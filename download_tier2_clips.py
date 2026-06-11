#!/usr/bin/env python3
"""
download_tier2_clips.py

Downloads free public-domain video clips from Wikimedia Commons for use
as Tier 2 VLM Oracle input (A3 camera / A4 spatial / A5 epistemic cells).

No login, no API key, no registration required.

Usage:
    python3 download_tier2_clips.py
    python3 download_tier2_clips.py --out-dir data/clips --n-clips 40 --max-mb 80
    python3 download_tier2_clips.py --build-manifest   # recover URLs for already-downloaded clips

Output:
    data/clips/{filename}.webm  (or .mp4)
    data/clips/manifest.json    {stem: {url, title, size_mb}} — used by eval / download scripts
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

UA = "SycoBench-Vid/1.0 (academic research)"
API = "https://commons.wikimedia.org/w/api.php"

# Categories chosen to provide varied content for each Tier 2 axis.
# A3 (camera motion): sports, outdoor action
# A4 (spatial layout changes): cooking, team activity, dance
# A5 (epistemic/quality): archival, nature, mixed quality
CATEGORIES = [
    # A3 — camera motion, tracking shots, handheld footage
    "Category:Videos of ball sports",
    "Category:Videos of athletics",
    "Category:Videos of swimming",
    "Category:Videos of cycling",
    # A4 — spatial arrangement changes, people/objects moving in frame
    "Category:Videos of cooking",
    "Category:Videos of dance",
    "Category:Videos of people walking",
    "Category:Videos of football (soccer)",
    # A5 — varied visual quality, lighting, clarity
    "Category:NASA videos",
    "Category:Videos of wildlife",
    "Category:Videos of birds",
    "Category:Videos of weather",
]


def _api(params: dict) -> dict:
    url = API + "?" + "&".join(f"{k}={urllib.request.quote(str(v))}" for k, v in params.items())
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def get_video_files(category: str, limit: int = 20) -> list[dict]:
    """Return list of {title, url, size_mb} for video files in a category."""
    try:
        data = _api({
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": limit,
            "format": "json",
        })
    except Exception as e:
        print(f"  Warning: could not query {category}: {e}")
        return []

    titles = [
        m["title"] for m in data.get("query", {}).get("categorymembers", [])
        if any(m["title"].lower().endswith(ext) for ext in (".webm", ".mp4", ".ogv"))
    ]
    if not titles:
        return []

    # Fetch download URLs and sizes in one call
    try:
        info = _api({
            "action": "query",
            "titles": "|".join(titles[:10]),  # API limit
            "prop": "imageinfo",
            "iiprop": "url|size",
            "format": "json",
        })
    except Exception as e:
        print(f"  Warning: imageinfo failed for {category}: {e}")
        return []

    results = []
    for page in info.get("query", {}).get("pages", {}).values():
        ii = page.get("imageinfo", [{}])[0]
        url  = ii.get("url", "")
        size = ii.get("size", 0)
        title = page.get("title", "")
        if url and size > 0:
            results.append({
                "title": title,
                "url": url,
                "size_mb": size / 1e6,
                "stem": Path(url).stem[:60].replace(" ", "_").replace("/", "_"),
            })
    return results


def download(url: str, dest: Path, retries: int = 4) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                while chunk := r.read(65536):
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                print(f" rate-limited, waiting {wait}s...", end=" ", flush=True)
                time.sleep(wait)
            else:
                print(f" HTTP {e.code}")
                break
        except Exception as e:
            print(f" ERROR: {e}")
            break
    if dest.exists():
        dest.unlink()
    return False


MANIFEST_NAME = "manifest.json"


def load_manifest(out: Path) -> dict:
    mp = out / MANIFEST_NAME
    if mp.exists():
        return json.loads(mp.read_text())
    return {}


def save_manifest(out: Path, manifest: dict):
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))


def build_manifest_for_existing(out: Path, manifest: dict) -> dict:
    """Query Wikimedia API to recover URLs for already-downloaded clips missing from manifest."""
    existing_stems = {
        p.stem: p for p in out.iterdir()
        if p.suffix in (".webm", ".mp4", ".ogv") and p.stem not in manifest
    }
    if not existing_stems:
        return manifest

    print(f"Recovering URLs for {len(existing_stems)} existing clips not in manifest...")
    for stem, path in existing_stems.items():
        # Decode percent-encoding back to Wikimedia title
        title_guess = "File:" + urllib.request.unquote(stem).replace("_", " ") + path.suffix
        try:
            info = _api({
                "action": "query",
                "titles": title_guess,
                "prop": "imageinfo",
                "iiprop": "url|size",
                "format": "json",
            })
            for page in info.get("query", {}).get("pages", {}).values():
                ii = page.get("imageinfo", [{}])[0]
                url = ii.get("url", "")
                size = ii.get("size", 0)
                if url:
                    manifest[stem] = {"url": url, "title": title_guess, "size_mb": size / 1e6}
                    print(f"  Found: {stem[:50]}")
                    break
        except Exception as e:
            print(f"  Could not recover URL for {stem}: {e}")
        time.sleep(1)
    return manifest


def main():
    p = argparse.ArgumentParser(
        description="Download Wikimedia Commons clips for Tier 2 VLM Oracle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir",        default="data/clips")
    p.add_argument("--n-clips",        type=int, default=40, help="Target number of clips to download")
    p.add_argument("--max-mb",         type=float, default=80, help="Skip clips larger than this (MB)")
    p.add_argument("--min-mb",         type=float, default=0.5, help="Skip clips smaller than this (MB)")
    p.add_argument("--build-manifest", action="store_true",
                   help="Query Wikimedia to recover URLs for already-downloaded clips, then exit")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(out)

    if args.build_manifest:
        manifest = build_manifest_for_existing(out, manifest)
        save_manifest(out, manifest)
        print(f"Manifest saved: {len(manifest)} entries → {out / MANIFEST_NAME}")
        return

    already = {p.stem for p in out.iterdir() if p.suffix in (".webm", ".mp4", ".ogv")}
    print(f"Already have {len(already)} clips in {out}/")

    candidates = []
    seen_urls: set[str] = set()
    print(f"\nQuerying {len(CATEGORIES)} Wikimedia categories...")
    for cat in CATEGORIES:
        files = get_video_files(cat, limit=20)
        for f in files:
            if f["url"] not in seen_urls:
                seen_urls.add(f["url"])
                candidates.append(f)
        time.sleep(2)  # polite delay between category queries

    # Filter by size and not already downloaded
    usable = [
        c for c in candidates
        if args.min_mb <= c["size_mb"] <= args.max_mb and c["stem"] not in already
    ]
    print(f"Found {len(usable)} usable clips ({args.min_mb}–{args.max_mb} MB) not yet downloaded")
    usable.sort(key=lambda x: x["size_mb"])  # download smallest first

    downloaded = 0
    for c in usable:
        if downloaded >= args.n_clips:
            break
        suffix = Path(c["url"]).suffix.lower() or ".webm"
        dest = out / f"{c['stem']}{suffix}"
        print(f"[{downloaded+1}/{args.n_clips}] {c['stem'][:50]} ({c['size_mb']:.1f} MB) ...",
              end=" ", flush=True)
        if download(c["url"], dest):
            print("OK")
            downloaded += 1
            manifest[c["stem"]] = {"url": c["url"], "title": c["title"], "size_mb": c["size_mb"]}
            save_manifest(out, manifest)  # save after each download so progress survives interrupts
        time.sleep(5)   # polite delay — Wikimedia CDN rate-limits fast downloaders

    total = [f for f in out.iterdir() if f.suffix in (".webm", ".mp4", ".ogv")]
    print(f"\nDone. {downloaded} new clips downloaded. {len(total)} total clips in {out}/")
    print(f"Manifest: {len(manifest)} entries → {out / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
