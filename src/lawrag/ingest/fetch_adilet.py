"""Connected-side snapshot fetcher for ИПС «Әділет».

This module never runs on the air-gapped machine. It runs on an internet-facing
host, writes a bundle directory, and that directory is what physically crosses
the gap. Keeping the fetcher inside the same package is deliberate: the parser
and the fetcher have to agree on the document format, and splitting them across
repositories is how they drift.

The authoritative source of Kazakhstan legislation is the Эталонный контрольный
банк НПА maintained by the Institute of Legislation and Legal Information under
the Ministry of Justice; adilet.zan.kz is its public interface. If your
organisation can obtain a direct export from the reference bank, prefer it —
point ``--source-dir`` at the export and skip this fetcher entirely.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ..logging_setup import get_logger
from .acts import REGISTRY, ActSpec

log = get_logger(__name__)

BASE = "https://adilet.zan.kz"
LANG_PATH = {"ru": "rus", "kk": "kaz"}
DEFAULT_DELAY_S = 2.0


@dataclass(slots=True)
class FetchPlan:
    act: ActSpec
    lang: str
    url: str
    filename: str
    as_of: date | None = None


def build_plan(codes: list[str] | None = None, langs: tuple[str, ...] = ("ru", "kk"),
               as_of: date | None = None) -> list[FetchPlan]:
    selected = [REGISTRY[c] for c in (codes or REGISTRY.keys()) if c in REGISTRY]
    plans: list[FetchPlan] = []
    unverified = [s.code for s in selected if s.official_id and not s.verified]
    if unverified:
        log.warning(
            "document ids for %s have NOT been confirmed against a live URL. "
            "An unverified id yields a 404, not a wrong document, but confirm "
            "it at https://adilet.zan.kz/rus/index/docs before relying on it.",
            ", ".join(unverified),
        )
    for spec in selected:
        if not spec.official_id:
            log.warning("no official id registered for %s — skipping", spec.code)
            continue
        for lang in langs:
            url = f"{BASE}/{LANG_PATH[lang]}/docs/{spec.official_id}"
            if as_of:
                # adilet serves point-in-time redactions from the same path.
                url = f"{url}?ysclid=&date={as_of.isoformat()}"
            plans.append(FetchPlan(spec, lang, url, f"{spec.code}.{lang}.html", as_of))
    return plans


def _build_ssl_verify(ca_bundle: str | None = None, insecure: bool = False):
    """Return an httpx ``verify`` value (an ``ssl.SSLContext`` or ``False``).

    httpx verifies against certifi's bundle and never consults the operating
    system trust store. A certificate signed by a root that the OS trusts but
    certifi does not ship — a corporate/antivirus TLS proxy, or the Kazakhstan
    national root that fronts some government services — therefore fails with
    "unable to get local issuer certificate". Preferring the OS store fixes
    that without pinning any particular root here.

    Note: httpx>=0.28 accepts only ``bool`` or ``ssl.SSLContext`` for ``verify``
    (no path strings), so every branch below returns a context, not a path.
    """
    import ssl

    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        log.warning("TLS verification DISABLED (--insecure). Snapshots are not "
                    "authenticated in transit; only use this on a trusted network.")
        return ctx

    if ca_bundle:
        log.info("verifying TLS against CA bundle %s", ca_bundle)
        return ssl.create_default_context(cafile=ca_bundle)

    try:
        import truststore

        log.info("verifying TLS against the OS trust store (truststore)")
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ModuleNotFoundError:
        import certifi

        log.info("truststore not installed; verifying TLS against certifi %s. If a "
                 "fetch fails with a certificate error, mount the missing root and "
                 "pass --ca-bundle PATH.",
                 getattr(certifi, "__version__", "?"))
        return ssl.create_default_context(cafile=certifi.where())


def _is_cert_error(exc: BaseException) -> bool:
    """True if *exc* (or any cause in its chain) is a TLS trust failure."""
    import ssl

    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, ssl.SSLCertVerificationError):
            return True
        msg = str(e)
        if "CERTIFICATE_VERIFY_FAILED" in msg or "unable to get local issuer" in msg:
            return True
        e = e.__cause__ or e.__context__
    return False


def _is_redaction_as_of(html: str, as_of: date) -> bool:
    """Check that a point-in-time page really is that point in time.

    A redaction in force on a date cannot cite an amending law *signed* after
    that date. Commencement dates are no use for this check: a current page
    legitimately cites laws already adopted that commence next year. This is
    the only evidence available that the server honoured the date parameter
    rather than quietly returning today's text under a historic label.
    """
    from . import structure as S
    from .dating import parse_amendments
    from .parser import adilet_html_to_lines

    for line in adilet_html_to_lines(html):
        if not S.is_footnote_line(line):
            continue
        if any(a.law_date > as_of for a in parse_amendments(line)):
            return False
    return True


_SHELL_MARKERS = ("включите JavaScript", "JavaScript қосыңыз", "enable JavaScript")


def is_app_shell(html: str) -> bool:
    """True for the page adilet serves to clients that do not run JavaScript.

    Since September 2026 the act routes return a small shell with the act's
    title and a note that the full text is deliberately absent for such
    clients (its robots policy closes the site to corpus-collecting crawlers
    and keeps it open to search and citation). Before this check the fetcher
    saved that shell as the document and reported success, and the next
    ingest parsed a code with zero articles.
    """
    lowered = html.lower()
    if any(m.lower() in lowered for m in _SHELL_MARKERS):
        return True
    from . import structure as S
    from .parser import adilet_html_to_lines

    lines = adilet_html_to_lines(html)
    lang = S.detect_lang("\n".join(lines))
    return not any(S.match_heading(line, lang) for line in lines)


def fetch(plans: list[FetchPlan], out_dir: Path, delay_s: float = DEFAULT_DELAY_S,
          timeout_s: float = 60.0, ca_bundle: str | None = None,
          insecure: bool = False) -> list[Path]:
    """Download each planned document. Polite by default; this is a public service."""
    import httpx

    verify = _build_ssl_verify(ca_bundle, insecure)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    meta: list[dict] = []
    cert_hint_shown = False

    with httpx.Client(timeout=timeout_s, follow_redirects=True, verify=verify,
                      headers={"User-Agent": "lawrag-corpus-sync/0.1"}) as client:
        for i, plan in enumerate(plans, 1):
            log.info("[%d/%d] %s", i, len(plans), plan.url)
            try:
                resp = client.get(plan.url)
                resp.raise_for_status()
            except Exception as exc:  # noqa: BLE001 — one bad act must not kill the sync
                log.error("failed %s: %s", plan.url, exc)
                if _is_cert_error(exc) and not cert_hint_shown:
                    cert_hint_shown = True
                    log.error(
                        "TLS trust failure: adilet's certificate could not be verified "
                        "against the trust store in use. The signing root is almost "
                        "certainly trusted by your OS but missing from certifi (a "
                        "corporate/antivirus TLS proxy, or the Kazakhstan national "
                        "root). Hand the root in: pass --ca-bundle PATH "
                        "(or LAWRAG_CA_BUNDLE); failing that, pass --insecure to skip "
                        "verification (last resort, trusted network only)."
                    )
                continue
            if is_app_shell(resp.text):
                log.error(
                    "%s: adilet returned its JavaScript shell, not the act. The site now "
                    "withholds act text from non-browser clients on purpose; obtain the "
                    "text from the reference bank export (--source-dir) instead. Not saved.",
                    plan.url)
                continue
            if plan.as_of and not _is_redaction_as_of(resp.text, plan.as_of):
                log.error(
                    "%s: adilet returned a text amended after %s — the date parameter was "
                    "not honoured, so this is not the redaction asked for. Not saved.",
                    plan.url, plan.as_of)
                continue
            path = out_dir / plan.filename
            path.write_text(resp.text, encoding="utf-8")
            written.append(path)
            meta.append({
                "file": plan.filename,
                "code": plan.act.code,
                "kind": plan.act.kind.value,
                "official_id": plan.act.official_id,
                "lang": plan.lang,
                "title": plan.act.title_ru if plan.lang == "ru" else plan.act.title_kk,
                "url": plan.url,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                # The day this text is known to be in force. The parser dates
                # each article from its footnotes and never past this day.
                **({"as_of": plan.as_of.isoformat()} if plan.as_of else {}),
            })
            time.sleep(delay_s)

    # Merge into an existing index rather than replacing it. Re-fetching one
    # code into data/adilet used to rewrite documents.json with that code only,
    # and the next ingest silently dropped every other act in the directory.
    index_path = out_dir / "documents.json"
    merged: dict[str, dict] = {}
    if index_path.exists():
        try:
            for entry in json.loads(index_path.read_text(encoding="utf-8")):
                if (out_dir / entry["file"]).exists():
                    merged[entry["file"]] = entry
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("existing %s is unreadable (%s); rewriting it", index_path, exc)
    for entry in meta:
        merged[entry["file"]] = entry
    index_path.write_text(
        json.dumps(list(merged.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("fetched %d/%d documents into %s", len(written), len(plans), out_dir)
    return written
