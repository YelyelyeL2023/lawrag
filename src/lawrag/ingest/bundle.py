"""Corpus snapshot bundles — the only thing that crosses the air gap.

A bundle is a directory of source documents plus a manifest listing every file
with its SHA-256. Import verifies the manifest before a single byte is parsed,
so a truncated copy or a tampered file fails loudly instead of silently becoming
"the law" inside the system. The bundle id is derived from the manifest hash,
which makes every indexed chunk traceable to an exact, reproducible snapshot.

Signing is intentionally left as a hook rather than a bundled crypto stack: an
air-gapped deployment already has an established key-handling procedure, and
this should defer to it. ``verify_signature`` is where that plugs in.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..logging_setup import get_logger

log = get_logger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass(slots=True)
class ManifestEntry:
    path: str
    sha256: str
    bytes: int


@dataclass(slots=True)
class Manifest:
    source: str
    created_at: str
    entries: list[ManifestEntry]
    note: str = ""

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for e in sorted(self.entries, key=lambda x: x.path):
            h.update(f"{e.path}:{e.sha256}:{e.bytes}\n".encode())
        return h.hexdigest()

    @property
    def release_id(self) -> str:
        return f"{self.created_at[:10]}-{self.digest[:12]}"


def sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def build_manifest(root: Path, source: str, note: str = "") -> Manifest:
    entries: list[ManifestEntry] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == MANIFEST_NAME:
            continue
        digest, size = sha256_file(path)
        entries.append(ManifestEntry(str(path.relative_to(root)), digest, size))
    return Manifest(
        source=source,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        entries=entries,
        note=note,
    )


def write_manifest(root: Path, manifest: Manifest) -> Path:
    target = root / MANIFEST_NAME
    payload = {
        "source": manifest.source,
        "created_at": manifest.created_at,
        "note": manifest.note,
        "digest": manifest.digest,
        "release_id": manifest.release_id,
        "entries": [asdict(e) for e in manifest.entries],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def read_manifest(root: Path) -> Manifest:
    payload = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    return Manifest(
        source=payload["source"],
        created_at=payload["created_at"],
        entries=[ManifestEntry(**e) for e in payload["entries"]],
        note=payload.get("note", ""),
    )


class BundleVerificationError(RuntimeError):
    pass


def verify_bundle(root: Path, *, require_signature: bool = False) -> Manifest:
    """Verify every file against the manifest. Raises on any mismatch."""
    manifest = read_manifest(root)
    stated = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if stated.get("digest") != manifest.digest:
        raise BundleVerificationError(
            "manifest digest does not match its own entries — the manifest was edited"
        )

    problems: list[str] = []
    listed = {e.path for e in manifest.entries}
    for entry in manifest.entries:
        path = root / entry.path
        if not path.exists():
            problems.append(f"missing: {entry.path}")
            continue
        digest, size = sha256_file(path)
        if digest != entry.sha256:
            problems.append(f"checksum mismatch: {entry.path}")
        elif size != entry.bytes:
            problems.append(f"size mismatch: {entry.path}")

    for path in root.rglob("*"):
        if path.is_file() and path.name != MANIFEST_NAME:
            rel = str(path.relative_to(root))
            if rel not in listed and not rel.endswith(".sig"):
                problems.append(f"unlisted file present: {rel}")

    if require_signature and not verify_signature(root, manifest):
        problems.append("signature missing or invalid")

    if problems:
        raise BundleVerificationError(
            f"bundle {root} failed verification:\n  " + "\n  ".join(problems)
        )
    log.info("bundle %s verified (%d files, release %s)",
             root, len(manifest.entries), manifest.release_id)
    return manifest


def verify_signature(root: Path, manifest: Manifest) -> bool:
    """Hook for the site's signing procedure.

    Expected to validate ``manifest.json.sig`` against the organisation's
    trusted key. Returns False when no signature is present so that callers
    passing ``require_signature=True`` fail closed.
    """
    sig = root / f"{MANIFEST_NAME}.sig"
    if not sig.exists():
        return False
    log.warning(
        "signature file present but no verifier is configured — "
        "wire verify_signature() to your key infrastructure before production"
    )
    return False


def pack(source_dir: Path, out_dir: Path, source: str, note: str = "") -> Manifest:
    """Assemble a bundle directory ready to carry across the air gap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            dest = out_dir / path.relative_to(source_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    manifest = build_manifest(out_dir, source=source, note=note)
    write_manifest(out_dir, manifest)
    log.info("packed %d files into %s (release %s)",
             len(manifest.entries), out_dir, manifest.release_id)
    return manifest
