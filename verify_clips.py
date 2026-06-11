#!/usr/bin/env python3
"""
verify_clips.py

Queries Wikimedia Commons categories for .webm/.mp4 video files,
verifies each download URL is accessible, and saves the first
TARGET_CLIPS verified entries to verified_clips.json.

Run this ONCE on any internet-connected machine before handing off
the benchmark build to the GPU machine. No GPU required.

Usage:
    python verify_clips.py
    python verify_clips.py --target 500 --out verified_clips.json

Resumes automatically if interrupted — already-verified URLs are kept.
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

UA  = "SycoBench-Vid/2.0 (academic research)"
API = "https://commons.wikimedia.org/w/api.php"

TARGET_CLIPS = 500
MIN_MB       = 1.0
MAX_MB       = 150.0

# ── Wikimedia categories known to have .webm content ──────────────────────────
# Curated from live queries — each yields 10–50+ webm files.
CATEGORIES = [
    # Sports & movement (good for A1 subject, A3 camera, A4 spatial)
    "Category:Videos of athletics",
    "Category:Videos of swimming",
    "Category:Videos of cycling",
    "Category:Videos of dance",
    "Category:Videos of gymnastics",
    "Category:Videos of martial arts",
    "Category:Videos of rowing",
    "Category:Videos of skiing",
    "Category:Videos of surfing",
    "Category:Videos of skateboarding",
    "Category:Videos of parkour",
    "Category:Videos of acrobatics",
    "Category:Videos of juggling",
    "Category:Videos of circus",
    "Category:Videos of weightlifting",
    "Category:Videos of boxing",
    "Category:Videos of volleyball",
    "Category:Videos of tennis",
    "Category:Videos of basketball",
    "Category:Videos of sailing",
    "Category:Videos of football",
    "Category:Videos of rugby",
    "Category:Videos of baseball",
    "Category:Videos of cricket",
    "Category:Videos of hockey",
    "Category:Videos of archery",
    "Category:Videos of fencing",
    "Category:Videos of equestrian sports",
    # Craft & activity (good for A1, A2 indoor scene, A4 spatial)
    "Category:Videos of cooking",
    "Category:Videos of crafts",
    "Category:Videos of baking",
    "Category:Videos of woodworking",
    "Category:Videos of blacksmithing",
    "Category:Videos of glassblowing",
    "Category:Videos of pottery",
    "Category:Videos of textile arts",
    "Category:Videos of traditional crafts",
    "Category:Videos of food preparation",
    "Category:Videos of street food",
    "Category:Videos of gardening",
    "Category:Videos of agriculture",
    "Category:Videos of fishing",
    "Category:Videos of construction",
    "Category:Videos of demolition",
    # Visual phenomena (good for A5 epistemic, A3 camera)
    "Category:Time-lapse videos",
    "Category:Stop motion videos",
    "Category:Slow motion videos",
    "Category:Underwater videos",
    "Category:Aerial videos",
    "Category:Videos of fireworks",
    "Category:Videos of fire",
    "Category:Videos of water",
    "Category:Videos of volcanoes",
    "Category:Videos of floods",
    "Category:Videos of lightning",
    "Category:Videos of snow",
    "Category:Videos of fog",
    "Category:Videos of sunsets",
    "Category:Videos of auroras",
    # Animals (good for A1 subject, A2 background/habitat)
    "Category:Videos of animals",
    "Category:Videos of birds",
    "Category:Videos of dogs",
    "Category:Videos of cats",
    "Category:Videos of horses",
    "Category:Videos of insects",
    "Category:Videos of fish",
    "Category:Videos of marine animals",
    "Category:Videos of primates",
    "Category:Videos of reptiles",
    # Scenes & environment (good for A2 background, A5 quality)
    "Category:Videos of street scenes",
    "Category:Videos of markets",
    "Category:Videos of parades",
    "Category:Videos of festivals",
    "Category:Videos of ceremonies",
    "Category:Videos of protests",
    "Category:Videos of fairs",
    "Category:Videos of concerts",
    "Category:Videos of performances",
    "Category:Videos of lectures",
    "Category:Videos of interviews",
    "Category:Videos of trains",
    "Category:Videos of aircraft",
    "Category:Videos of ships",
    "Category:Videos of cars",
    "Category:Videos of motorcycles",
    # Arts (good for diverse visual content)
    "Category:Videos of performing arts",
    "Category:Videos of folklore",
    "Category:Videos of traditional dance",
    "Category:Videos of traditional music",
    "Category:Videos of opera",
    "Category:Videos of ballet",
    "Category:Videos of circus arts",
    "Category:Videos of magic",
    "Category:Videos of puppetry",
    "Category:Videos of visual arts",
    "Category:Videos of sculpture techniques",
    "Category:Videos of street art",
]


# ── API helpers ────────────────────────────────────────────────────────────────

def _api(params: dict, retries: int = 4) -> dict:
    url = API + "?" + "&".join(
        f"{k}={urllib.request.quote(str(v), safe='')}" for k, v in params.items()
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    Rate-limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5)
    return {}


def get_category_files(category: str, min_mb: float, max_mb: float) -> list[dict]:
    """
    Get all .webm/.mp4 files in a category (pages through cmcontinue).
    Returns list of {title, url, size_mb, stem}.
    """
    results    = []
    continue_  = None
    seen_titles: set[str] = set()

    while True:
        params = {
            "action":  "query",
            "list":    "categorymembers",
            "cmtitle": category,
            "cmtype":  "file",
            "cmlimit": 50,
            "format":  "json",
        }
        if continue_:
            params["cmcontinue"] = continue_

        data    = _api(params)
        members = data.get("query", {}).get("categorymembers", [])

        # Filter to webm/mp4 by title extension
        titles = [
            m["title"] for m in members
            if m["title"].lower().endswith((".webm", ".mp4"))
            and m["title"] not in seen_titles
        ]
        seen_titles.update(titles)

        if titles:
            # Fetch download URLs + sizes in batches of 10 (API limit)
            for i in range(0, len(titles), 10):
                batch = titles[i:i+10]
                try:
                    info = _api({
                        "action":  "query",
                        "titles":  "|".join(batch),
                        "prop":    "imageinfo",
                        "iiprop":  "url|size",
                        "format":  "json",
                    })
                    for page in info.get("query", {}).get("pages", {}).values():
                        ii   = page.get("imageinfo", [{}])[0]
                        url  = ii.get("url", "")
                        size = ii.get("size", 0) / 1_000_000
                        if url and min_mb <= size <= max_mb:
                            results.append({
                                "url":     url,
                                "title":   page.get("title", ""),
                                "size_mb": round(size, 2),
                                "stem":    Path(url).stem[:100],
                            })
                    time.sleep(1)
                except Exception as e:
                    print(f"    imageinfo error: {e}")

        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        continue_ = cont
        time.sleep(1.5)

    return results


def verify_url(url: str, timeout: int = 10) -> bool:
    """HEAD request with one 429 retry. Fails fast on everything else."""
    for attempt in range(2):
        try:
            time.sleep(1.0)   # per-thread pacing — keeps CDN happy across 3 threads
            req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status == 200
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(15)  # single short backoff on rate-limit, then retry
            else:
                return False
        except Exception:
            return False
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Pre-verify Wikimedia video URLs for SycoBench-Vid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", type=int, default=TARGET_CLIPS)
    p.add_argument("--out",    default="verified_clips.json")
    p.add_argument("--min-mb", type=float, default=MIN_MB)
    p.add_argument("--max-mb", type=float, default=MAX_MB)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-verified clips
    verified: list[dict]    = []
    verified_urls: set[str] = set()
    verified_stems: set[str] = set()
    if out_path.exists():
        try:
            verified       = json.loads(out_path.read_text())
            verified_urls  = {c["url"]  for c in verified}
            verified_stems = {c["stem"] for c in verified}
            print(f"Resuming — {len(verified)} clips already verified.")
        except Exception:
            pass

    if len(verified) >= args.target:
        print(f"Already have {len(verified)} verified clips. Nothing to do.")
        return

    print(f"Target: {args.target} verified clips")
    print(f"Size filter: {args.min_mb}–{args.max_mb} MB\n")

    # Cache of all discovered-but-not-yet-verified candidates — persists across restarts
    candidates_path = out_path.parent / "candidates_cache.json"
    candidates: list[dict] = []
    seen_urls: set[str]    = set(verified_urls)

    if candidates_path.exists():
        try:
            cached = json.loads(candidates_path.read_text())
            candidates = [c for c in cached
                          if c["url"] not in seen_urls and c["stem"] not in verified_stems]
            seen_urls.update(c["url"] for c in candidates)
            print(f"Loaded {len(candidates)} candidates from cache (skipping API queries).")
        except Exception:
            candidates = []

    if not candidates:
        print(f"Querying {len(CATEGORIES)} Wikimedia categories...")
        for i, cat in enumerate(CATEGORIES):
            if len(verified) + len(candidates) >= args.target * 3:
                break
            try:
                files = get_category_files(cat, args.min_mb, args.max_mb)
                new   = [f for f in files if f["url"] not in seen_urls
                                          and f["stem"] not in verified_stems]
                for f in new:
                    seen_urls.add(f["url"])
                candidates.extend(new)
                print(f"  [{i+1:>3}/{len(CATEGORIES)}] {cat.split(':')[1]:<45} "
                      f"+{len(new):>3} new  (pool={len(candidates)})")
            except Exception as e:
                print(f"  [{i+1:>3}/{len(CATEGORIES)}] {cat.split(':')[1]:<45} ERROR: {e}")
            time.sleep(2)

        candidates_path.write_text(json.dumps(candidates, indent=2))
        print(f"Saved {len(candidates)} candidates to cache.\n")

    # Verify all candidates in parallel
    save_lock = Lock()
    needed    = args.target - len(verified)
    print(f"Verifying {len(candidates)} candidates with 5 threads (need {needed} more)...\n")

    def _check(c):
        return c, verify_url(c["url"])

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(_check, c) for c in candidates]
        done = 0
        for fut in as_completed(futures):
            if len(verified) >= args.target:
                break
            c, ok = fut.result()
            done += 1
            label  = c["stem"][:52]
            status = "OK  " if ok else "FAIL"
            print(f"  [{len(verified)+1:>3}/{args.target}] {status}  "
                  f"{label:<52} {c['size_mb']:>6.1f} MB  "
                  f"(checked {done}/{len(candidates)})")
            if ok:
                with save_lock:
                    verified.append(c)
                    verified_urls.add(c["url"])
                    verified_stems.add(c["stem"])
                    out_path.write_text(json.dumps(verified, indent=2))

    print(f"{'='*60}")
    print(f"Done. {len(verified)} verified clips → {out_path}")
    if len(verified) < args.target:
        print(f"WARNING: only {len(verified)}/{args.target} clips found.")
        print("More categories may be needed — edit CATEGORIES list in verify_clips.py.")


if __name__ == "__main__":
    main()
