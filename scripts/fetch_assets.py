"""Fetch (and convert) the assets that are NOT committed to this repo.

Today that is exactly one: ANYmal-C. Its upstream URDF package (ANYbotics,
BSD-3-Clause) is downloaded at a pinned commit, then converted to USD with the
engine's URDF converter into `assets/anymal_converted/` — the path
`AnymalConfig.robot` points at.

Usage:
    python scripts/fetch_assets.py            # fetch + convert everything missing
    python scripts/fetch_assets.py --check    # exit 0 if all fetched assets present
    python scripts/fetch_assets.py --force    # re-download + re-convert

The download step needs only the network. The convert step needs the engine
(LUMENGINE_ROOT) and the converter environment the engine already uses
(urdf_usd_converter in the system Python, or LM_RL_CONVERTER_PYTHON).
"""
import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets"

# Pinned upstream: github.com/ANYbotics/anymal_c_simple_description @ master
# on 2026-07-16. Bump deliberately; the commit is part of the asset identity.
ANYMAL_REPO = "ANYbotics/anymal_c_simple_description"
ANYMAL_COMMIT = "87b68511622f0f17e78d9ddca7d862f150a93fb4"
ANYMAL_SRC_DIR = ASSETS / "anymal_c_simple_description"     # downloaded package
ANYMAL_OUT_DIR = ASSETS / "anymal_converted"                # converter output
ANYMAL_OUT_USD = ANYMAL_OUT_DIR / "anymal.usda"             # AnymalConfig.robot


def _rel(path: Path) -> str:
    """Repo-relative display path; absolute when outside the repo (e.g. tests)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _download_zip(repo: str, commit: str) -> zipfile.ZipFile:
    url = f"https://github.com/{repo}/archive/{commit}.zip"
    print(f"[fetch] downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    return zipfile.ZipFile(io.BytesIO(data))


def fetch_anymal_source(force: bool) -> Path:
    """Download the ANYbotics description package into assets/. Returns the URDF path."""
    if ANYMAL_SRC_DIR.exists() and not force:
        print(f"[fetch] {ANYMAL_SRC_DIR.name} already present (use --force to refresh)")
    else:
        if ANYMAL_SRC_DIR.exists():
            shutil.rmtree(ANYMAL_SRC_DIR)
        zf = _download_zip(ANYMAL_REPO, ANYMAL_COMMIT)
        tmp = ASSETS / f".{ANYMAL_SRC_DIR.name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        zf.extractall(tmp)
        # The zip wraps everything in <repo>-<commit>/ — unwrap it.
        (inner,) = [p for p in tmp.iterdir() if p.is_dir()]
        inner.rename(ANYMAL_SRC_DIR)
        shutil.rmtree(tmp, ignore_errors=True)
        # Record provenance next to the payload.
        (ANYMAL_SRC_DIR / "PROVENANCE.md").write_text(
            f"Fetched by scripts/fetch_assets.py from github.com/{ANYMAL_REPO} "
            f"@ {ANYMAL_COMMIT} (BSD-3-Clause, (c) ANYbotics AG). Do not edit.\n",
            encoding="utf-8")
        print(f"[fetch] extracted -> {ANYMAL_SRC_DIR}")
    urdfs = sorted(ANYMAL_SRC_DIR.rglob("*.urdf"))
    if not urdfs:
        raise RuntimeError(f"no .urdf found under {ANYMAL_SRC_DIR}")
    # Prefer the plain anymal.urdf if present.
    for u in urdfs:
        if u.stem == "anymal":
            return u
    return urdfs[0]


def convert_anymal(urdf: Path, force: bool):
    """Convert the URDF to USD into assets/anymal_converted/ via the engine."""
    if ANYMAL_OUT_USD.exists() and not force:
        print(f"[convert] {_rel(ANYMAL_OUT_USD)} already present")
        return
    from lumengine_envs import assets as _assets
    __import__('lumengine_envs._engine', fromlist=['x']).ensure_engine()          # raises with a clear message if LUMENGINE_ROOT unset
    from lm.rl import _convert      # engine converter wrapper (subprocess, clean PYTHONPATH)

    print(f"[convert] {_rel(urdf)} -> {_rel(ANYMAL_OUT_DIR)}")
    produced = Path(_convert.convert_urdf(str(urdf)))
    src_dir = produced.parent
    if ANYMAL_OUT_DIR.exists():
        shutil.rmtree(ANYMAL_OUT_DIR)
    shutil.copytree(src_dir, ANYMAL_OUT_DIR)
    got = ANYMAL_OUT_DIR / produced.name
    if got != ANYMAL_OUT_USD:
        got.rename(ANYMAL_OUT_USD)   # AnymalConfig.robot expects anymal.usda
    print(f"[convert] wrote {_rel(ANYMAL_OUT_USD)}")


def check() -> int:
    ok = True
    for label, path in [("ANYmal-C converted USD", ANYMAL_OUT_USD)]:
        present = path.exists()
        print(f"[check] {'OK  ' if present else 'MISS'} {label}: {_rel(path)}")
        ok &= present
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="report presence, don't fetch")
    ap.add_argument("--force", action="store_true", help="re-download and re-convert")
    ap.add_argument("--no-convert", action="store_true",
                    help="download only (skip the engine-dependent USD conversion)")
    args = ap.parse_args()

    if args.check:
        sys.exit(check())

    urdf = fetch_anymal_source(args.force)
    if args.no_convert:
        print("[fetch] --no-convert: stopping after download")
        return
    convert_anymal(urdf, args.force)
    sys.exit(check())


if __name__ == "__main__":
    main()
