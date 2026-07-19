"""Asset management — resolution, download cache, and the fetch CLI
(`lumotion-fetch-assets`).

Resolution order (`assets_dir()`):
1. LUMOTION_ASSETS environment variable (explicit override);
2. the repo checkout's assets/ next to this package (dev / git-clone flow);
3. the download cache `%LOCALAPPDATA%/lumotion-envs/assets-<ref>/assets`,
   filled by `lumotion-fetch-assets` for wheel installs (no checkout).

The asset pack is the `assets/` tree of the public lm-envs repo, downloaded
as a pinned GitHub archive — the ref is part of the asset identity, so a
future lumotion-envs release pins the matching tag. On top of the pack, the
ANYmal-C sources (not redistributed — fetched from ANYbotics at a pinned
commit) are downloaded and converted to USD with the engine.
"""
import argparse
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# The asset-pack pin. "main" during development; release tags of
# lumotion-envs bump this to the matching repo tag.
ASSETS_REF = os.environ.get("LUMOTION_ASSETS_REF", "main")
ASSETS_REPO = "Lumengine/lm-envs"

# Pinned upstream: github.com/ANYbotics/anymal_c_simple_description @ master
# on 2026-07-16. Bump deliberately; the commit is part of the asset identity.
ANYMAL_REPO = "ANYbotics/anymal_c_simple_description"
ANYMAL_COMMIT = "87b68511622f0f17e78d9ddca7d862f150a93fb4"


def _cache_root() -> Path:
    local = os.environ.get("LOCALAPPDATA", str(Path.home()))
    return Path(local) / "lumotion-envs"


def _pack_dir(ref: str = ASSETS_REF) -> Path:
    return _cache_root() / f"assets-{ref}"


def assets_dir() -> Path:
    env = os.environ.get("LUMOTION_ASSETS")
    if env:
        return Path(env)
    repo_assets = Path(__file__).resolve().parents[1] / "assets"
    if repo_assets.is_dir():
        return repo_assets
    return _pack_dir() / "assets"


ASSETS = assets_dir()


def _download_zip(repo: str, ref: str) -> zipfile.ZipFile:
    url = f"https://github.com/{repo}/archive/{ref}.zip"
    print(f"[fetch] downloading {url}")
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    print(f"[fetch] {len(data) / (1 << 20):.0f} MB")
    return zipfile.ZipFile(io.BytesIO(data))


def fetch_pack(force: bool = False, ref: str = ASSETS_REF) -> Path:
    """Download the repo's assets/ tree into the versioned cache. Idempotent."""
    dest = _pack_dir(ref)
    manifest = dest / "manifest.json"
    if manifest.exists() and not force:
        print(f"[fetch] asset pack already cached at {dest} (use --force to refresh)")
        return dest / "assets"

    zf = _download_zip(ASSETS_REPO, ref)
    tmp = dest.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    # Extract only <repo>-<ref>/assets/** — the rest of the archive is code.
    members = [m for m in zf.namelist() if "/assets/" in m and not m.endswith("/")]
    if not members:
        raise RuntimeError(f"archive of {ASSETS_REPO}@{ref} has no assets/ tree")
    for m in members:
        rel = Path(*Path(m).parts[1:])          # drop the <repo>-<ref>/ wrapper
        out = tmp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src, open(out, "wb") as f:
            shutil.copyfileobj(src, f)
    (tmp / "manifest.json").write_text(json.dumps({
        "source": f"github.com/{ASSETS_REPO}", "ref": ref,
        "files": len(members),
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1), encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    print(f"[fetch] asset pack ({len(members)} files) -> {dest}")
    return dest / "assets"


def fetch_anymal_source(assets: Path, force: bool = False) -> Path:
    """Download the ANYbotics description package (BSD-3-Clause, fetched — not
    redistributed by us). Returns the URDF path."""
    src_dir = assets / "anymal_c_simple_description"
    if src_dir.exists() and not force:
        print(f"[fetch] {src_dir.name} already present (use --force to refresh)")
    else:
        if src_dir.exists():
            shutil.rmtree(src_dir)
        zf = _download_zip(ANYMAL_REPO, ANYMAL_COMMIT)
        tmp = assets / f".{src_dir.name}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        zf.extractall(tmp)
        (inner,) = [p for p in tmp.iterdir() if p.is_dir()]
        inner.rename(src_dir)
        shutil.rmtree(tmp, ignore_errors=True)
        (src_dir / "PROVENANCE.md").write_text(
            f"Fetched by lumotion-fetch-assets from github.com/{ANYMAL_REPO} "
            f"@ {ANYMAL_COMMIT} (BSD-3-Clause, (c) ANYbotics AG). Do not edit.\n",
            encoding="utf-8")
        print(f"[fetch] extracted -> {src_dir}")
    urdfs = sorted(src_dir.rglob("*.urdf"))
    if not urdfs:
        raise RuntimeError(f"no .urdf found under {src_dir}")
    for u in urdfs:
        if u.stem == "anymal":
            return u
    return urdfs[0]


def convert_anymal(assets: Path, urdf: Path, force: bool = False) -> None:
    """Convert the ANYmal URDF to USD via the engine (wheel or LUMENGINE_ROOT)."""
    out_dir = assets / "anymal_converted"
    out_usd = out_dir / "anymal.usda"           # AnymalConfig.robot expects this
    if out_usd.exists() and not force:
        print(f"[convert] {out_usd} already present")
        return
    from lumotion_envs._engine import ensure_engine
    ensure_engine()                              # raises with a clear remedy
    from lm.rl import _convert                   # engine converter wrapper

    print(f"[convert] {urdf} -> {out_dir}")
    produced = Path(_convert.convert_urdf(str(urdf)))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(produced.parent, out_dir)
    got = out_dir / produced.name
    if got != out_usd:
        got.rename(out_usd)
    print(f"[convert] wrote {out_usd}")


# Sentinels: one vendored asset per family + the fetched/converted ANYmal.
_CHECKS = [
    ("Cartpole USD", "cartpole_converted/cartpole.usda"),
    ("Ant MJCF", "ant.xml"),
    ("Go2 URDF", "go2/urdf/go2.urdf"),
    ("Franka MJCF", "franka/panda.xml"),
    ("ANYmal-C converted USD", "anymal_converted/anymal.usda"),
]


def check(assets: Path) -> int:
    ok = True
    for label, rel in _CHECKS:
        present = (assets / rel).exists()
        print(f"[check] {'OK  ' if present else 'MISS'} {label}: {assets / rel}")
        ok &= present
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="Fetch the Lumotion asset pack (and the non-redistributed "
                    "ANYmal-C sources) into place.")
    ap.add_argument("--check", action="store_true", help="report presence, don't fetch")
    ap.add_argument("--force", action="store_true", help="re-download and re-convert")
    ap.add_argument("--no-convert", action="store_true",
                    help="download only (skip the engine-dependent USD conversion)")
    args = ap.parse_args()

    assets = assets_dir()
    if args.check:
        sys.exit(check(assets))

    # Wheel install (no checkout, no override): populate the versioned cache.
    repo_assets = Path(__file__).resolve().parents[1] / "assets"
    if not os.environ.get("LUMOTION_ASSETS") and not repo_assets.is_dir():
        assets = fetch_pack(args.force)

    urdf = fetch_anymal_source(assets, args.force)
    if args.no_convert:
        print("[fetch] --no-convert: stopping after download")
        return
    convert_anymal(assets, urdf, args.force)
    sys.exit(check(assets))


if __name__ == "__main__":
    main()
