#!/usr/bin/env python3
"""
Analysis for all layers of the V5 pipeline. Same conventions as the earlier
package's analysis, carried forward deliberately: cluster-bootstrap CIs over
video_id (not raw proportion CIs — records from the same clip aren't
independent), every rate reported with n and CI, nulls framed as "excludes
effects bigger than the CI" rather than "ruled out."

Tries a proper logistic regression via statsmodels if it's installed;
degrades gracefully to binned/stratified comparisons (still real evidence,
just coarser) if not, rather than failing outright or silently skipping.
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def cluster_boot(recs, key_fn, val_fn, n=4000, seed=0):
    byclip = defaultdict(list)
    for r in recs:
        byclip[key_fn(r)].append(r)
    clips = list(byclip)
    if not clips:
        return None, None
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = [x for c in rng.choices(clips, k=len(clips)) for x in byclip[c]]
        v = val_fn(s)
        if v is not None:
            out.append(v)
    if not out:
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def flip_rate(recs):
    vals = [r["prob_flip"] for r in recs if r.get("prob_flip") is not None]
    return sum(vals) / len(vals) if vals else None


def flip_ci(recs, key_fn=lambda r: (r["source"], r["video_id"])):
    m = flip_rate(recs)
    if m is None:
        return "n=0"
    lo, hi = cluster_boot(recs, key_fn, flip_rate)
    lo_s = f"{100*lo:.1f}" if lo is not None else "?"
    hi_s = f"{100*hi:.1f}" if hi is not None else "?"
    return f"{100*m:.1f}% [{lo_s}, {hi_s}]  (n={len(recs)})"


def initially_correct(rows):
    return [r for r in rows if r.get("argmax_pre") == r.get("correct_index")]


def section(title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")


def analyze_probe_regression(out_dir, model):
    rows = load_jsonl(Path(out_dir) / f"results_probe_{model}.jsonl")
    if not rows:
        return
    section("Layer 1+3 — behavioral probe + continuous regression")
    val = initially_correct(rows)
    print(f"Total records: {len(rows)}, initially-correct: {len(val)}")
    print(f"Overall flip rate: {flip_ci(val)}")

    print("\n-- flip rate by pre-pressure margin (quintiles; the confidence-alone baseline) --")
    margins = sorted(r["margin_pre"] for r in val if r.get("margin_pre") is not None)
    if margins:
        edges = np.quantile(margins, np.linspace(0, 1, 6))
        for i in range(5):
            lo_e, hi_e = edges[i], edges[i + 1]
            bucket = [r for r in val if r.get("margin_pre") is not None and lo_e <= r["margin_pre"] <= hi_e]
            print(f"  margin in [{lo_e:.2f}, {hi_e:.2f}]: {flip_ci(bucket)}")

    for source in sorted({r["source"] for r in val}):
        sub = [r for r in val if r["source"] == source]
        print(f"\n-- {source}: flip rate by reasoning_tag --")
        for tag in sorted({r.get("reasoning_tag") for r in sub if r.get("reasoning_tag")}):
            print(f"  {tag:<20}: {flip_ci([r for r in sub if r.get('reasoning_tag') == tag])}")

    pt = [r for r in val if r["source"] == "perception_test"]
    if pt:
        print("\n-- perception_test: flip rate by author-flagged distractor tag --")
        with_d = [r for r in pt if r.get("has_distractor_tag") is True]
        without_d = [r for r in pt if r.get("has_distractor_tag") is False]
        print(f"  has_distractor_tag=True : {flip_ci(with_d)}")
        print(f"  has_distractor_tag=False: {flip_ci(without_d)}")
        print("  (this is the measured, author-provided version of the 'plausible foil' axis — "
              "no hand-constructed near/far distractor sets involved)")

        fracs = [(r["clip_action_fraction"], r["prob_flip"]) for r in pt
                 if r.get("clip_action_fraction") is not None and r.get("prob_flip") is not None]
        if fracs:
            arr = np.array(fracs)
            corr = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
            print(f"\n  corr(clip_action_fraction, prob_flip) = {corr:.3f} (n={len(fracs)}) — "
                  "per-CLIP covariate, not per-question; interpret as a coarse signal only.")

    try:
        import statsmodels.api as sm
        import pandas as pd
        df = pd.DataFrame(val)
        df = df.dropna(subset=["margin_pre", "prob_flip"])
        df["source_pt"] = (df["source"] == "perception_test").astype(int)
        X = sm.add_constant(df[["margin_pre", "source_pt"]])
        y = df["prob_flip"].astype(int)
        model_fit = sm.Logit(y, X).fit(disp=0)
        print("\n-- logistic regression: prob_flip ~ margin_pre + source (statsmodels available) --")
        print(model_fit.summary2().tables[1])
    except ImportError:
        print("\n(statsmodels not installed — showing stratified breakdowns above only; "
              "pip install statsmodels for a proper mixed/logistic fit)")
    except Exception as e:
        print(f"\n(logistic regression failed: {e} — stratified breakdowns above still stand)")


def analyze_grounding(out_dir, model):
    rows = load_jsonl(Path(out_dir) / f"grounding_summary_{model}.jsonl")
    if not rows:
        return
    section("Layer 2 — grounding-alignment (real NExT-GQA windows)")
    print(f"Items captured: {len(rows)}")
    ratios_pre = [r["ratio_pre"] for r in rows]
    ratios_post = [r["ratio_post"] for r in rows]
    print(f"mean ratio_pre  (attention-in-window / chance): {np.mean(ratios_pre):.2f} "
          f"(median {np.median(ratios_pre):.2f})")
    print(f"mean ratio_post (attention-in-window / chance): {np.mean(ratios_post):.2f} "
          f"(median {np.median(ratios_post):.2f})")
    print("ratio > 1 means attention concentrates on the real annotated evidence window more "
          "than chance; ratio ~= 1 means attention doesn't track it at all.")

    flipped = [r for r in rows if r.get("prob_flip")]
    held = [r for r in rows if r.get("prob_flip") is False]
    if flipped and held:
        print(f"\nflipped items (n={len(flipped)}): mean ratio_pre={np.mean([r['ratio_pre'] for r in flipped]):.2f}")
        print(f"held items    (n={len(held)}): mean ratio_pre={np.mean([r['ratio_pre'] for r in held]):.2f}")
        print("(does weaker real-evidence grounding pre-pressure predict who flips? descriptive "
              "only at this n — no significance test run here.)")


def analyze_dilution(out_dir, model):
    rows = load_jsonl(Path(out_dir) / f"results_dilution_{model}.jsonl")
    if not rows:
        return
    section("Layer 4 — token-count dilution")
    val = initially_correct(rows)
    conds = sorted({r["condition"] for r in val},
                   key=lambda c: (c.startswith("static"), int("".join(filter(str.isdigit, c)) or 0)))
    for cond in conds:
        print(f"  {cond:>10}: {flip_ci([r for r in val if r['condition'] == cond])}")
    print("\nCompare static-N vs real-N at matched N for the token-count-vs-content contrast; "
          "compare across N within either arm for the pure token-count trend.")


def analyze_mech(out_dir, model, kind, layer_key):
    rows = load_jsonl(Path(out_dir) / f"{kind}_{model}.jsonl")
    if not rows:
        return
    section(f"Layer 5 — {kind}")
    by_layer = defaultdict(list)
    for r in rows:
        k = tuple(r[layer_key]) if isinstance(r[layer_key], list) else r[layer_key]
        by_layer[k].append(r["restored"])
    best = None
    for k in sorted(by_layer, key=lambda k: (k if isinstance(k, tuple) else (k,))):
        vals = by_layer[k]
        rate = sum(vals) / len(vals)
        print(f"  {layer_key}={k}: restored {sum(vals)}/{len(vals)} ({100*rate:.1f}%)")
        if best is None or rate > best[1]:
            best = (k, rate)
    if best:
        n_items = len({r["video_id"] for r in rows})
        print(f"\nPeak restoration at {layer_key}={best[0]} ({100*best[1]:.1f}%), n={n_items} items. "
              "Causal claim (patching/knockout actually changes output), but treat the specific "
              "layer as a hypothesis for replication, not a settled fact — and remember this "
              "tests the final readout, not live generation (see script docstrings).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--model", choices=["qwen25", "qwen3"], default="qwen3")
    args = ap.parse_args()

    analyze_probe_regression(args.out_dir, args.model)
    analyze_grounding(args.out_dir, args.model)
    analyze_dilution(args.out_dir, args.model)
    analyze_mech(args.out_dir, args.model, "knockout", "window")
    analyze_mech(args.out_dir, args.model, "patching", "layer")

    section("Done")
    print("Keep n/CI attached to every rate above when writing this up. Grounding-alignment and "
          "dilution results are observational except the Layer 5 knockout/patching rows, which "
          "are the only causal claims in this pipeline.")


if __name__ == "__main__":
    main()
