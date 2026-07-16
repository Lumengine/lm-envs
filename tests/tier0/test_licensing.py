"""Tier 0 — the legal manifest matches what is actually shipped.

Rules enforced:
1. Every vendored robot asset directory carries its upstream LICENSE file.
2. Every vendored directory has a row in THIRD_PARTY_LICENSES.md whose status
   is `vendored` (a dir present while its row says `planned` is a manifest bug
   — exactly what happened with Allegro).
3. Generated/converted dirs (gitignored or produced by fetch_assets) are exempt.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets"
MANIFEST = REPO / "THIRD_PARTY_LICENSES.md"

# asset dir -> keyword that must appear in its manifest row.
VENDORED = {
    "a1": "Unitree A1",
    "go1": "Unitree A1",          # shares the "A1 / Go1" row
    "go2": "Unitree Go2",
    "h1": "Unitree H1",
    "franka": "Franka",
    "allegro": "Allegro",
    "sektion_cabinet": "Sektion",
}
# Converted/fetched outputs (gitignored, produced by scripts/fetch_assets.py or
# the converter) — no LICENSE requirement of their own.
EXEMPT_DIRS = {"anymal_converted", "anymal_c_simple_description", "cartpole_converted"}


def _manifest_rows():
    rows = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|.*\|\s*(vendored|fetched|planned)\s*\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


def test_manifest_parses():
    rows = _manifest_rows()
    assert len(rows) >= 10, "THIRD_PARTY_LICENSES.md table not found or reformatted"


@pytest.mark.parametrize("dirname", sorted(VENDORED), ids=str)
def test_vendored_dir_has_license_file(dirname):
    d = ASSETS / dirname
    assert d.is_dir(), f"assets/{dirname} missing"
    assert (d / "LICENSE").exists(), f"assets/{dirname}/LICENSE missing"


@pytest.mark.parametrize("dirname,keyword", sorted(VENDORED.items()), ids=lambda x: str(x))
def test_vendored_dir_has_vendored_row(dirname, keyword):
    rows = _manifest_rows()
    matches = {asset: status for asset, status in rows.items() if keyword.lower() in asset.lower()}
    assert matches, f"no THIRD_PARTY_LICENSES.md row mentions {keyword!r} (for assets/{dirname})"
    assert "vendored" in matches.values(), (
        f"assets/{dirname} is committed but its manifest row says {matches} — "
        f"update THIRD_PARTY_LICENSES.md")


def test_no_unknown_asset_dirs():
    """Every committed asset dir is either declared vendored here or exempt —
    forces this test (and the manifest) to grow with the catalog."""
    dirs = {p.name for p in ASSETS.iterdir() if p.is_dir()}
    unknown = dirs - set(VENDORED) - EXEMPT_DIRS
    assert not unknown, (
        f"asset dirs not covered by the licensing test: {sorted(unknown)} — "
        f"add them to VENDORED (+ manifest row) or EXEMPT_DIRS")
