"""
The unified item schema both dataset loaders (perception_test_items.py,
nextqa_items.py) produce. Every downstream script (probe, grounding-check,
dilution, mechanistic) consumes this dict shape and nothing dataset-specific,
so adding a third dataset later only means writing one more loader.

Required keys, always present:
    source            "perception_test" | "nextqa"
    video_id          str
    video_path        str — resolved path to the mp4 (may not exist yet if
                       videos haven't been downloaded; check before using)
    qid               str | int
    question          str, verbatim from the dataset
    options           list[str], verbatim from the dataset (3-way for
                       Perception Test, 5-way for NExT-QA) — never
                       hand-constructed by this pipeline
    correct_index     int, index into `options`
    reasoning_tag      str | None — dataset's own difficulty/type label
                       (Perception Test: descriptive/explanatory/predictive/
                       counterfactual. NExT-QA: causal_why/causal_how/
                       temporal_next/temporal_current/temporal_previous/
                       descriptive_*). Different vocabularies — do not pool
                       across datasets as if equivalent; keep `source` as a
                       covariate in any pooled regression.

Optional / dataset-specific (None when not applicable — check before use):
    clip_duration_s        float | None
    area_tag                str | None (Perception Test only)
    content_tags            list[str] (Perception Test only; e.g.
                             "distractor object", "sequencing")
    has_distractor_tag      bool | None — True if the dataset authors flagged
                             a deliberate plausible foil (Perception Test
                             only; None, not False, for NExT-QA — absence of
                             the tag there means "not annotated", not
                             "confirmed absent")
    clip_action_count       int | None (Perception Test only, per-CLIP not
                             per-question)
    clip_action_fraction    float | None (Perception Test only, per-CLIP)
    type_code               str | None (NExT-QA only, raw CW/CH/TN/... code)
    clip_frame_count         int | None (NExT-QA only)
    grounding_windows       list[(start_s, end_s), ...] | None — REAL
                             per-question temporal grounding, only ever
                             populated for NExT-QA val/test items that
                             NExT-GQA covers (~68% of val, ~65% of test, 0%
                             of train). None means "not available", not
                             "no relevant evidence".
"""

REQUIRED_KEYS = {"source", "video_id", "video_path", "qid", "question",
                 "options", "correct_index", "reasoning_tag"}


def validate_item(item):
    """Cheap sanity check before spending GPU time on an item. Raises
    AssertionError with a specific message rather than failing deep inside
    a probe call with a confusing KeyError."""
    missing = REQUIRED_KEYS - item.keys()
    assert not missing, f"item missing required keys: {missing}"
    assert isinstance(item["options"], list) and len(item["options"]) >= 2, \
        f"item {item['source']}/{item['video_id']}/{item['qid']} has <2 options"
    assert 0 <= item["correct_index"] < len(item["options"]), \
        f"item {item['source']}/{item['video_id']}/{item['qid']} correct_index out of range"
    return True
