"""
Loads NExT-QA (+ NExT-GQA grounding, where available) into the unified Item
schema. Options and questions are verbatim from the dataset authors — no
hand-constructed distractors.

Verified against the real bundled files: NExT-QA's CSVs give 5-way MCQ
(a0-a4), human-authored, with a `type` code (CW/CH causal-why/how, TN/TC/TP
temporal-next/current/previous, DC/DL/DO descriptive-count/location/other) —
use this as the reasoning-tag covariate, analogous to Perception Test's
`reasoning` field but a different vocabulary (kept as its own field, not
merged, since pooling the two datasets' tag vocabularies as if equivalent
would be exactly the kind of unjustified "these are comparable" move this
project has been trying to avoid).

NExT-GQA's gsub_{val,test}.json gives REAL per-question temporal grounding —
keyed video_id -> qid -> list of [start_s, end_s] windows — but only for the
val/test splits (grounding annotation is expensive; train has none). This is
the dataset to use for Layer 2 (grounding-alignment), not Perception Test,
which has no per-question grounding at all (see perception_test_items.py).
"""
import csv, json
from pathlib import Path

TYPE_LABELS = {
    "CW": "causal_why", "CH": "causal_how",
    "TN": "temporal_next", "TC": "temporal_current", "TP": "temporal_previous",
    "DC": "descriptive_count", "DL": "descriptive_location", "DO": "descriptive_other",
}


def load_map_vid_vidorid(annotations_dir):
    return json.load(open(Path(annotations_dir) / "map_vid_vidorID.json"))


def load_grounding(annotations_dir, split):
    path = Path(annotations_dir) / f"gsub_{split}.json"
    if not path.exists():
        return {}
    return json.load(open(path))


def resolve_video_path(video_dir, vid, vidor_map):
    """NExT-QA's extracted zip nests videos under a subfolder path from
    map_vid_vidorID.json (e.g. video 2909445186 -> 0101/2909445186.mp4).
    Falls back to a flat <video_dir>/<vid>.mp4 in case the archive was
    re-flattened during download/extraction."""
    video_dir = Path(video_dir)
    subpath = vidor_map.get(vid)
    if subpath:
        candidate = video_dir / f"{subpath}.mp4"
        if candidate.exists():
            return str(candidate)
    flat = video_dir / f"{vid}.mp4"
    return str(flat)


def build_items(annotations_dir, video_dir, split="val"):
    """Yields unified-schema item dicts for one split. split="val" or "test"
    are the only ones with any NExT-GQA grounding; "train" has QA only."""
    annotations_dir = Path(annotations_dir)
    csv_path = annotations_dir / f"{split}.csv"
    rows = list(csv.DictReader(open(csv_path)))
    vidor_map = load_map_vid_vidorid(annotations_dir)
    grounding = load_grounding(annotations_dir, split)

    for r in rows:
        vid = r["video_id"]
        qid = r["qid"]
        options = [r[f"a{i}"] for i in range(5)]
        try:
            correct_index = options.index(r["answer"])
        except ValueError:
            # answer text doesn't exact-match an option string (rare CSV quirk) — skip rather
            # than guess which option was intended.
            continue
        frame_count = int(r["frame_count"]) if r.get("frame_count") else None

        windows = None
        vground = grounding.get(vid)
        if vground:
            loc = vground.get("location", {}).get(qid)
            if loc:
                windows = [tuple(w) for w in loc]
            duration_s = vground.get("duration")
        else:
            duration_s = None

        type_code = r.get("type", "")
        yield {
            "source": "nextqa",
            "video_id": vid,
            "video_path": resolve_video_path(video_dir, vid, vidor_map),
            "qid": qid,
            "question": r["question"],
            "options": options,
            "correct_index": correct_index,
            "clip_duration_s": duration_s,          # only known where NExT-GQA covers this video
            "clip_frame_count": frame_count,
            "reasoning_tag": TYPE_LABELS.get(type_code, type_code),
            "type_code": type_code,
            "area_tag": None,
            "content_tags": [],
            "has_distractor_tag": None,              # not annotated in NExT-QA; leave unknown, don't guess
            "clip_action_count": None,
            "clip_action_fraction": None,
            "grounding_windows": windows,
        }


def summarize(annotations_dir, split="val"):
    items = list(build_items(annotations_dir, video_dir=".", split=split))
    from collections import Counter
    type_ctr = Counter(it["type_code"] for it in items)
    n_grounded = sum(1 for it in items if it["grounding_windows"])
    return {
        "n_items": len(items),
        "n_videos": len({it["video_id"] for it in items}),
        "type_counts": dict(type_ctr),
        "n_with_grounding": n_grounded,
    }


if __name__ == "__main__":
    import sys, json as _json
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    print(_json.dumps(summarize(sys.argv[1], split), indent=2))
