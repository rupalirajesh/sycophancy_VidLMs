"""
Loads Perception Test annotations into the unified Item schema (see
datasets/item_schema.py). No hand-constructed distractors anywhere here —
every question and option comes verbatim from the dataset's own authors.

IMPORTANT SCOPE NOTE, found by inspecting the real annotation file (not
assumed): Perception Test's `mc_question` entries are NOT linked to a specific
`action_localisation` time window — there is no shared id between them. The
`grounded_question` field looked promising but its "answers" are
`object_tracking` ids (spatial identity), not timestamps. So Perception Test
items carry NO per-question temporal grounding — do not claim they do.

What Perception Test items DO carry, verified against real data:
- exact clip duration and a real `action_localisation` list (median action
  span ~7.8% of clip duration, ~6.25 actions/clip) usable as a per-CLIP
  (not per-question) continuous covariate — "how much of this clip is
  labeled activity," "how many competing actions," etc.
- author-assigned `reasoning` (descriptive/explanatory/predictive/
  counterfactual) and `area` (physics/semantics/abstraction/memory) tags
  per question.
- `tag` list per question, including "distractor object"/"distractor action"
  when the authors deliberately built in a plausible foil — this is the
  measured-not-guessed version of the "plausibility" axis the Charades-based
  E5 experiment tried to hand-construct with near/far scene clusters.

For per-question temporal grounding (Layer 2, "does attention track the exact
evidence window"), use NExT-GQA instead (nextqa_items.py) — it's built for
exactly that; Perception Test is not.
"""
import json
from pathlib import Path


def load_annotations(annotations_path):
    return json.load(open(annotations_path))


def build_items(annotations_path, video_dir, min_options=2):
    """Yields unified-schema item dicts. video_dir should contain
    video_<id>.mp4 files (as produced by download_perception_test.py)."""
    data = load_annotations(annotations_path)
    video_dir = Path(video_dir)

    for vid, rec in data.items():
        meta = rec.get("metadata", {})
        fps = meta.get("frame_rate") or 0
        num_frames = meta.get("num_frames") or 0
        duration_s = (num_frames / fps) if fps else None
        video_path = video_dir / f"{vid}.mp4"

        actions = rec.get("action_localisation", [])
        action_count = len(actions)
        action_frac = None
        if duration_s and actions:
            total = sum((a["timestamps"][1] - a["timestamps"][0]) / 1e6 for a in actions)
            action_frac = min(total / duration_s, 1.0)

        for q in rec.get("mc_question", []):
            options = q.get("options", [])
            if len(options) < min_options:
                continue
            tags = q.get("tag", []) or []
            yield {
                "source": "perception_test",
                "video_id": vid,
                "video_path": str(video_path),
                "qid": q["id"],
                "question": q["question"],
                "options": options,
                "correct_index": q["answer_id"],
                "clip_duration_s": duration_s,
                "reasoning_tag": q.get("reasoning"),
                "area_tag": q.get("area"),
                "content_tags": tags,
                "has_distractor_tag": any("distractor" in t for t in tags),
                "clip_action_count": action_count,
                "clip_action_fraction": action_frac,   # per-CLIP covariate, not per-question
                "grounding_windows": None,              # Perception Test does not provide this
            }


def summarize(annotations_path):
    """Quick counts for sanity-checking a downloaded annotation file."""
    items = list(build_items(annotations_path, video_dir="."))
    from collections import Counter
    reasoning_ctr = Counter(it["reasoning_tag"] for it in items)
    n_distractor = sum(1 for it in items if it["has_distractor_tag"])
    return {
        "n_items": len(items),
        "n_videos": len({it["video_id"] for it in items}),
        "reasoning_tag_counts": dict(reasoning_ctr),
        "n_with_distractor_tag": n_distractor,
    }


if __name__ == "__main__":
    import sys, json as _json
    print(_json.dumps(summarize(sys.argv[1]), indent=2))
