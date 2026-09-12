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
            plans.append(FetchPlan(spec, lang, url, f"{spec.code}.{lang}.html"))
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
                        "root). A container does not inherit the host's trust store, "
                        "so hand the root in: mount the PEM and pass --ca-bundle PATH "
                        "(or LAWRAG_CA_BUNDLE); failing that, pass --insecure to skip "
                        "verification (last resort, trusted network only)."
                    )
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
            })
            time.sleep(delay_s)

    (out_dir / "documents.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("fetched %d/%d documents into %s", len(written), len(plans), out_dir)
    return written
