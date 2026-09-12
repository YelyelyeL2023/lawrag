"""Bundle verification — the air-gap boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lawrag.ingest.bundle import (BundleVerificationError, build_manifest, pack,
                                  read_manifest, verify_bundle, write_manifest)


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.act.txt").write_text("alpha", encoding="utf-8")
    (src / "b.act.txt").write_text("beta", encoding="utf-8")
    out = tmp_path / "bundle"
    pack(src, out, source="test")
    return out


def test_clean_bundle_verifies(bundle):
    manifest = verify_bundle(bundle)
    assert len(manifest.entries) == 2
    assert manifest.release_id


def test_release_id_is_reproducible(tmp_path, bundle):
    first = read_manifest(bundle).digest
    again = build_manifest(bundle, source="test").digest
    assert first == again


def test_tampered_file_is_rejected(bundle):
    (bundle / "a.act.txt").write_text("alpha tampered", encoding="utf-8")
    with pytest.raises(BundleVerificationError, match="checksum mismatch"):
        verify_bundle(bundle)


def test_unlisted_file_is_rejected(bundle):
    (bundle / "smuggled.act.txt").write_text("gamma", encoding="utf-8")
    with pytest.raises(BundleVerificationError, match="unlisted file"):
        verify_bundle(bundle)


def test_missing_file_is_rejected(bundle):
    (bundle / "b.act.txt").unlink()
    with pytest.raises(BundleVerificationError, match="missing"):
        verify_bundle(bundle)


def test_edited_manifest_is_rejected(bundle):
    payload = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    payload["entries"][0]["sha256"] = "0" * 64
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BundleVerificationError, match="manifest was edited"):
        verify_bundle(bundle)


def test_signature_required_but_absent_fails_closed(bundle):
    with pytest.raises(BundleVerificationError, match="signature"):
        verify_bundle(bundle, require_signature=True)
