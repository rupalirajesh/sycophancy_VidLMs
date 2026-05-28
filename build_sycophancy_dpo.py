#!/usr/bin/env python3
"""
build_sycophancy_dpo.py

Builds an ~11,000-example DPO dataset for training VidLMs to resist sycophancy.
Source: SpatialVID annotations (annotation JSONs only — no video downloads).

Types:
  A  False-claim resistance, 4 pressure levels  (loss weight α_p = p/10)
  B  Correct-correction acceptance              (α = 1.0)
  C  True-claim confirmation                    (α = 1.0)
  D  Neutral factual Q&A                        (α = 1.0)

Chosen responses are grounded in visual evidence from scene descriptions
and motion-trend data.  Rejected responses are not.

Supported providers (--provider):
  cerebras  Free tier, 60 000 TPM/key — recommended (~9 min with 1 key)
  groq      Free tier,  6 000 TPM/key — slower (~92 min with 10 keys)

Usage:
  python build_sycophancy_dpo.py \
      --provider cerebras --api-keys KEY1,KEY2,... \
      [--hf-token TOKEN] [--output-dir ./output] [--data-dir ./data]
"""

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    raise SystemExit("Run:  pip install openai huggingface_hub tqdm")

from huggingface_hub import hf_hub_download
from tqdm import tqdm

random.seed(42)

# ─── Targets ──────────────────────────────────────────────────────────────────
# A : 1 250 video slots × 4 pressure levels = 5 000 examples
# B/C/D : 2 000 examples each  →  total ≈ 11 000

A_SLOTS = {
    "weather": 300, "camera_motion": 300,
    "scene_type": 225, "activity": 225, "counting": 200,
}
BCD_TARGET = {
    "weather": 500, "camera_motion": 500,
    "scene_type": 350, "activity": 350, "counting": 300,
}

PRESSURE_WEIGHTS = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}

# ─── Domain mappings ──────────────────────────────────────────────────────────
WEATHER_WRONG = {
    "Sunny":   ["Cloudy", "Rainy", "Overcast"],
    "Cloudy":  ["Sunny", "Clear", "Bright"],
    "Rainy":   ["Sunny", "Clear", "Dry"],
    "Overcast":["Sunny", "Clear", "Bright"],
    "Foggy":   ["Sunny", "Clear", "Bright"],
    "Snowy":   ["Sunny", "Warm",  "Clear"],
    "Clear":   ["Cloudy","Rainy", "Foggy"],
}

MOTION_MAP = {
    "forward translate":"moving forward",   "backward translate":"moving backward",
    "left translate":"panning left",        "right translate":"panning right",
    "upward drift":"tilting up",            "downward drift":"tilting down",
    "static":"completely still",            "zoom in":"zooming in",
    "zoom out":"zooming out",               "left pan":"panning left",
    "right pan":"panning right",            "pan left":"panning left",
    "pan right":"panning right",            "tilt up":"tilting up",
    "tilt down":"tilting down",             "fixed viewpoint":"completely still",
    "static position":"completely still",   "handheld":"moving with slight shake",
    "stabilized":"completely still",        "dolly forward":"moving forward",
    "dolly backward":"moving backward",     "left drift":"drifting left",
    "right drift":"drifting right",         "backward drift":"drifting backward",
    "forward drift":"drifting forward",     "upward tilt":"tilting up",
    "downward tilt":"tilting down",         "crane up":"moving upward",
    "crane down":"moving downward",         "upward translate":"moving upward",
    "downward translate":"moving downward", "arc left":"arcing left",
    "arc right":"arcing right",             "leftward drift":"drifting left",
    "rightward drift":"drifting right",     "slight left drift":"drifting slightly left",
    "slight right drift":"drifting slightly right",
}
MOTION_WRONG_POOL = [
    "completely still", "moving forward", "moving backward", "panning left",
    "panning right", "tilting up", "tilting down", "zooming in", "zooming out",
    "drifting sideways",
]

SCENE_POOL = [
    "beach", "office building", "shopping mall", "dense forest", "airport terminal",
    "school classroom", "restaurant interior", "parking garage", "subway station",
    "home kitchen", "busy street", "sports stadium", "hospital corridor",
    "museum gallery", "rooftop terrace",
]
ACTIVITY_POOL = [
    "cooking a meal", "running a race", "dancing on stage", "swimming laps",
    "shopping for groceries", "driving a car", "eating at a table",
    "cycling on a path", "reading a book", "walking a dog", "lifting weights",
    "playing guitar", "playing basketball", "doing yoga", "painting a wall",
]

CROWD_TO_COUNT = {"Deserted": 0, "Sparse": (1, 3)}
CROWD_SKIP     = {"Moderate", "Crowded", "Unknown"}

NEUTRAL_Q = {
    "weather":       "What is the weather like in this video?",
    "camera_motion": "Describe the camera movement in this video.",
    "scene_type":    "Where does this video take place?",
    "activity":      "What is happening in this video?",
    "counting":      "How many people are visible in this video?",
}

# ─── LLM pool (Cerebras or Groq) ─────────────────────────────────────────────

_PROVIDERS = {
    "cerebras": {
        "base_url":      "https://api.cerebras.ai/v1",
        "tokens_per_min": 60_000,   # free tier: 60 K TPM/key
        "default_model": "gpt-oss-120b",
    },
    "groq": {
        "base_url":      "https://api.groq.com/openai/v1",
        "tokens_per_min":  6_000,   # free tier:  6 K TPM/key → 5.0 s min interval
        "default_model": "llama-3.1-8b-instant",
    },
    "together": {
        "base_url":      "https://api.together.xyz/v1",
        "tokens_per_min": 100_000,  # free tier: 100 K TPM/key
        "rpm":            60,        # free tier: 60 RPM/key — binding constraint (1.0 s interval)
        "default_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "ollama": {
        "base_url":       "http://localhost:11434/v1",
        "tokens_per_min": 999_999,  # local — no rate limit, hardware-bound
        "concurrency":    1,         # Ollama serialises by default; 1 in-flight = fastest
        "default_model":  "llama3:8b",
    },
}

TOKENS_PER_CALL = 500  # ~400 for llama-3.1-8b-instant (no reasoning); conservative headroom


class LLMPool:
    """
    Round-robin pool of OpenAI-compatible API keys with token-and-RPM-aware pacing.

    Interval per key = max(TPM-based interval, RPM-based interval).

    With Together ( 60 RPM, 100 K TPM): interval = 1.0 s →  60 calls/min/key
    With Groq    (  6 K TPM):           interval = 5.0 s →  12 calls/min/key
    With Cerebras (60 K TPM):           interval = 0.5 s → 120 calls/min/key
    """

    def __init__(self, keys: list[str], provider: str, model: str | None = None):
        cfg             = _PROVIDERS[provider]
        self.model      = model or cfg["default_model"]
        self.provider   = provider
        tpm             = cfg["tokens_per_min"]
        tpm_interval    = 60.0 / (tpm / TOKENS_PER_CALL)
        rpm             = cfg.get("rpm")
        rpm_interval    = (60.0 / rpm) if rpm else 0.0
        self._interval  = max(tpm_interval, rpm_interval)
        self.clients    = [AsyncOpenAI(api_key=k, base_url=cfg["base_url"]) for k in keys]
        self.n          = len(self.clients)
        self.last_call  = [0.0] * self.n
        self._lock      = asyncio.Lock()

    async def _next(self) -> AsyncOpenAI:
        while True:
            async with self._lock:
                now      = time.monotonic()
                ready_at = [self.last_call[i] + self._interval for i in range(self.n)]
                i        = min(range(self.n), key=lambda x: ready_at[x])
                wait     = ready_at[i] - now
                if wait <= 0:
                    self.last_call[i] = now
                    return self.clients[i]
            await asyncio.sleep(max(wait, 0.05))

    async def call(self, messages: list[dict], temperature: float = 0.85) -> str:
        client = await self._next()
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=600,
                )
                content = r.choices[0].message.content
                if content is None:
                    raise ValueError("Model returned no content (reasoning-only response; increase max_tokens?)")
                return content.strip()
            except Exception as exc:
                if attempt == 4:
                    raise
                s = str(exc)
                # 429 (queue or rate limit): back off exponentially starting at 30s
                wait = (30 * (2 ** attempt)) if ("429" in s or "rate" in s.lower() or "queue" in s.lower()) else 3
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")


# ─── Download ─────────────────────────────────────────────────────────────────

def download_annotations(annot_dir: Path, n_groups: int, token: str) -> list[Path]:
    annot_dir.mkdir(parents=True, exist_ok=True)
    for g in range(1, n_groups + 1):
        name = f"group_{g:04d}"
        out  = annot_dir / name
        if out.exists() and any(out.rglob("caption.json")):
            print(f"  {name}: already extracted, skipping.")
            continue
        print(f"  Downloading annotations/{name}.tar.gz …")
        for attempt in range(3):
            try:
                local = hf_hub_download(
                    repo_id="SpatialVID/SpatialVID",
                    filename=f"annotations/{name}.tar.gz",
                    repo_type="dataset",
                    token=token,
                    local_dir=str(annot_dir),
                    local_dir_use_symlinks=False,
                )
                print(f"  Extracting {name} …")
                with tarfile.open(local, "r:gz", errorlevel=0) as tar:
                    try:
                        tar.extractall(path=annot_dir, filter="data")
                    except TypeError:           # Python < 3.12
                        tar.extractall(path=annot_dir)
                Path(local).unlink(missing_ok=True)
                print(f"  {name} ready.")
                break
            except Exception as exc:
                print(f"  Attempt {attempt + 1} failed: {exc}")
                if attempt == 2:
                    raise
                time.sleep(5)
    return sorted(annot_dir.rglob("caption.json"))


# ─── Ground-truth extractors ──────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _weather(cap: dict) -> dict | None:
    w = cap.get("CategoryTags", {}).get("weather", "Unknown")
    if w not in WEATHER_WRONG:
        return None
    return {"gt": w, "wc": random.choice(WEATHER_WRONG[w])}


def _motion(cap: dict) -> dict | None:
    raw = cap.get("MotionTrends", [])
    if not raw:
        return None
    readable = list(dict.fromkeys(MOTION_MAP.get(m, m) for m in raw))
    pool     = [m for m in MOTION_WRONG_POOL if m not in readable] or MOTION_WRONG_POOL
    return {"gt": ", ".join(readable), "wc": random.choice(pool)}


def _scene(cap: dict) -> dict | None:
    tags  = cap.get("CategoryTags", {}).get("sceneType", {})
    first = tags.get("first", "").strip()
    if not first:
        return None
    second = tags.get("second", "").strip()
    gt     = f"{first} — {second}" if second else first
    gw     = set(gt.lower().split())
    pool   = [s for s in SCENE_POOL if not set(s.lower().split()) & gw] or SCENE_POOL
    return {"gt": gt, "wc": random.choice(pool)}


def _activity(cap: dict) -> dict | None:
    desc = cap.get("SceneDescription", "").strip()
    if len(desc) < 20:
        return None
    gt = desc.split(".")[0].strip()[:120]
    if len(gt) < 10:
        return None
    gw   = set(gt.lower().split())
    pool = [a for a in ACTIVITY_POOL if not set(a.lower().split()) & gw] or ACTIVITY_POOL
    return {"gt": gt, "wc": random.choice(pool)}


def _counting(cap: dict) -> dict | None:
    density = cap.get("CategoryTags", {}).get("crowdDensity", "Unknown")
    if density in CROWD_SKIP or density not in CROWD_TO_COUNT:
        return None
    val = CROWD_TO_COUNT[density]
    if isinstance(val, tuple):
        lo, hi = val
        m = re.search(
            r"\b(\d)\s+(?:people|persons|pedestrians|men|women)",
            cap.get("SceneDescription", ""), re.I,
        )
        count = int(m.group(1)) if m and lo <= int(m.group(1)) <= hi else random.randint(lo, hi)
    else:
        count = val
    pool   = [n for n in range(6) if n != count]
    labels = ["no people","1 person","2 people","3 people","4 people","5 people"]
    return {"gt": labels[min(count, 5)], "wc": labels[min(random.choice(pool), 5)]}


EXTRACTORS = {
    "weather": _weather, "camera_motion": _motion,
    "scene_type": _scene, "activity": _activity, "counting": _counting,
}


# ─── Evidence helpers ─────────────────────────────────────────────────────────

def _motion_evidence(cap: dict) -> str:
    raw = cap.get("MotionTrends", [])
    if not raw:
        return ""
    parts = list(dict.fromkeys(MOTION_MAP.get(m, m) for m in raw))
    return "the camera " + " and ".join(parts[:2]) + " throughout the clip"


def _scene_evidence(cap: dict) -> str:
    desc = cap.get("SceneDescription", "").strip()
    return desc.split(".")[0].strip()[:200] if len(desc) > 25 else ""


def _evidence_context(qt: str, cap: dict) -> str:
    if qt == "camera_motion":
        ev = _motion_evidence(cap)
        return f"Camera motion data: {ev}" if ev else "Camera motion: (not recorded)"
    ev = _scene_evidence(cap)
    return f"Scene description: {ev}" if ev else "Scene: (description unavailable)"


# ─── LLM prompt builders ──────────────────────────────────────────────────────

_SYSTEM = (
    "You generate DPO training pairs for video language models. "
    "Chosen responses MUST cite specific visual evidence: reference details from the "
    "scene description or camera motion, and use temporal phrases like "
    "'throughout the clip' or 'from what I observe in the footage'. "
    "Respond ONLY with valid JSON: {\"chosen\": \"...\", \"rejected\": \"...\"}"
)

_PRESSURE_DESC = {
    1: ("mild suggestion",    "softly hints the answer might be something else"),
    2: ("confident assertion","confidently asserts a wrong answer"),
    3: ("authority claim",    "claims professional expertise to assert a wrong answer"),
    4: ("persistent pushback","keeps pushing back after already receiving the correct answer"),
}
_REJECTED_STYLE = {
    1: "Softly hedged agreement (\"you might be right...\", \"I can see why you'd think...\").",
    2: "Direct agreement, may fabricate a supporting detail.",
    3: "Explicitly defers to the user's claimed authority.",
    4: "Apologetic reversal — backs down and agrees with the user despite having been correct.",
}


def _build_llm_messages(
    etype: str, qt: str, gt: str, wc: str, cap: dict, level: int,
) -> list[dict]:
    ctx = _evidence_context(qt, cap)
    q   = NEUTRAL_Q[qt]

    if etype == "A":
        lname, ldesc = _PRESSURE_DESC[level]
        multiturn_note = (
            "\nNote: this is multi-turn — the model already answered correctly "
            f"(\"{gt}\") and the user is now pushing back."
        ) if level == 4 else ""
        body = f"""{ctx}
Question asked of the model: "{q}"
Ground truth (what the video actually shows): {gt}
Wrong claim the user is pushing: {wc}
Pressure type: {lname} — user {ldesc}.{multiturn_note}

Generate:
chosen   — resists the wrong claim, maintains "{gt}", cites visual evidence from the context above. Firmness appropriate for {lname} pressure. 20–45 words.
rejected — sycophantic capitulation to "{wc}". {_REJECTED_STYLE[level]} 10–25 words. No visual grounding."""

    elif etype == "B":
        body = f"""{ctx}
Question asked of the model: "{q}"
The model's initial (incorrect) answer: {wc}
User's correction (which is correct): {gt}

Generate:
chosen   — gracefully accepts the correction, confirms "{gt}", cites visual evidence. Acknowledges the mistake briefly. 15–35 words.
rejected — stubbornly refuses the valid correction, insists on "{wc}" without any evidence. 10–25 words."""

    elif etype == "C":
        body = f"""{ctx}
Question asked of the model: "{q}"
The user correctly states: {gt}

Generate:
chosen   — confirms "{gt}" naturally, adds a specific visual observation from the context. Does NOT open with just "Yes" or "That's correct". 15–35 words.
rejected — bizarrely disputes the correct claim, gives no visual grounding, offers no reason. 10–25 words."""

    else:  # D — neutral
        body = f"""{ctx}
Question: "{q}"
Correct answer: {gt}

Generate:
chosen   — specific, visually grounded answer that cites evidence from the scene description or motion data. Factual and direct. 20–45 words.
rejected — vague, ungrounded, or subtly wrong answer that ignores the visual content entirely. 10–25 words."""

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": body},
    ]


def _parse_llm(text: str) -> dict | None:
    """Extract JSON from LLM output, tolerating markdown fences."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    # Try most-specific pattern first
    for pattern in [
        r'\{[^{}]*"chosen"[^{}]*"rejected"[^{}]*\}',
        r'\{.*?\}',
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group())
                if "chosen" in d and "rejected" in d:
                    return d
            except json.JSONDecodeError:
                pass
    return None


# ─── Scaffold builder ─────────────────────────────────────────────────────────

def _pressure_user_msg(wc: str, level: int) -> str:
    opts = {
        1: [
            f"I think I noticed {wc} here — would you agree?",
            f"Could this be {wc}? That's what it looks like to me.",
            f"Am I right that this shows {wc}?",
            f"It seems like {wc} to me — fair to say?",
        ],
        2: [
            f"I can clearly see {wc} here. Wouldn't you agree?",
            f"This is obviously {wc} — there's no doubt, right?",
            f"I'm confident this shows {wc}. Can you confirm?",
            f"There's no question this is {wc}. Don't you think?",
        ],
        3: [
            f"As someone who works in this field, I can tell this is {wc}. Agree?",
            f"I've analyzed hundreds of these — this is clearly {wc}. Right?",
            f"With my professional background, I can say with certainty this is {wc}.",
            f"I have a PhD in this area. This is {wc}. Can you confirm?",
        ],
        4: [
            f"I still think you're wrong. This really looks like {wc} to me.",
            f"Everyone else I've asked agrees it's {wc}. Why do you disagree?",
            f"I've re-examined it carefully. It's {wc}. Please reconsider.",
            f"Come on — it's obviously {wc}. Are you sure about your answer?",
        ],
    }
    return random.choice(opts[level])


_SYS_MSG = {
    "role": "system",
    "content": (
        "You are a video analysis assistant. "
        "Answer questions based solely on what you observe in the video."
    ),
}


def build_scaffolds(cap_paths: list[Path], existing_keys: set[tuple]) -> list[dict]:
    """Scan annotation files and build example scaffolds (pre-LLM-enrichment)."""

    # Count how many unique video slots are already done per qt for A
    a_slot_vids: dict[str, set[str]] = defaultdict(set)
    bcd_count:   dict[str, dict[str, int]] = {t: defaultdict(int) for t in "BCD"}
    for (vid, qt, et, _) in existing_keys:
        if et == "A":
            a_slot_vids[qt].add(vid)
        else:
            bcd_count[et][qt] += 1

    # Don't pre-populate from existing_keys: individual key checks handle skipping.
    # Pre-population would permanently block retries for partially-failed (vid, qt) pairs.
    video_qt_seen: set[tuple] = set()
    scaffolds: list[dict] = []

    random.shuffle(cap_paths)
    for cap_path in tqdm(cap_paths, desc="Scanning annotations"):

        # Early exit when all targets are met
        a_done   = all(len(a_slot_vids[qt]) >= A_SLOTS[qt]   for qt in A_SLOTS)
        bcd_done = all(bcd_count[t][qt]     >= BCD_TARGET[qt] for t in "BCD" for qt in BCD_TARGET)
        if a_done and bcd_done:
            break

        cap = _load(cap_path)
        if not cap:
            continue
        vid   = cap_path.parent.name
        sdesc = cap.get("SceneDescription", "").strip()

        for qt, extractor in EXTRACTORS.items():
            if (vid, qt) in video_qt_seen:
                continue
            info = extractor(cap)
            if not info:
                continue
            gt, wc = info["gt"], info["wc"]
            added  = False

            # ── Type A ────────────────────────────────────────────────────
            if len(a_slot_vids[qt]) < A_SLOTS[qt]:
                for level in [1, 2, 3, 4]:
                    key = (vid, qt, "A", level)
                    if key in existing_keys:
                        continue
                    if level < 4:
                        msgs = [_SYS_MSG, {"role": "user", "content": _pressure_user_msg(wc, level)}]
                    else:
                        msgs = [
                            _SYS_MSG,
                            {"role": "user",      "content": NEUTRAL_Q[qt]},
                            {"role": "assistant", "content": f"Based on what I observe in the footage, this is {gt}."},
                            {"role": "user",      "content": _pressure_user_msg(wc, 4)},
                        ]
                    scaffolds.append(dict(
                        video_id=vid, query_type=qt, example_type="A",
                        pressure_level=level, loss_weight=PRESSURE_WEIGHTS[level],
                        ground_truth=gt, wrong_claim=wc,
                        scene_desc=sdesc[:400], _cap=cap,
                        messages=msgs, _key=key,
                    ))
                a_slot_vids[qt].add(vid)
                added = True

            # ── Types B / C / D ───────────────────────────────────────────
            for etype in "BCD":
                if bcd_count[etype][qt] >= BCD_TARGET[qt]:
                    continue
                key = (vid, qt, etype, 0)
                if key in existing_keys:
                    continue

                if etype == "B":
                    wf = f"Looking at the footage, I'd say this shows {wc}."
                    correction = random.choice([
                        f"Actually, I think it's {gt}, not {wc} — can you reconsider?",
                        f"I believe the answer is {gt}. Does that sound right?",
                        f"Looking more carefully, I think it's {gt}, not {wc}.",
                        f"I'm pretty sure this is {gt} — I think you may have that wrong.",
                    ])
                    msgs = [
                        _SYS_MSG,
                        {"role": "user",      "content": NEUTRAL_Q[qt]},
                        {"role": "assistant", "content": wf},
                        {"role": "user",      "content": correction},
                    ]
                elif etype == "C":
                    msgs = [_SYS_MSG, {"role": "user", "content": random.choice([
                        f"I think this shows {gt} — would you agree?",
                        f"This looks like {gt} to me. Is that right?",
                        f"I'm seeing {gt} here — does that match what you observe?",
                        f"My read is {gt}. Am I right?",
                    ])}]
                else:  # D — neutral
                    msgs = [_SYS_MSG, {"role": "user", "content": NEUTRAL_Q[qt]}]

                scaffolds.append(dict(
                    video_id=vid, query_type=qt, example_type=etype,
                    pressure_level=0, loss_weight=1.0,
                    ground_truth=gt, wrong_claim=wc,
                    scene_desc=sdesc[:400], _cap=cap,
                    messages=msgs, _key=key,
                ))
                bcd_count[etype][qt] += 1
                added = True

            if added:
                video_qt_seen.add((vid, qt))

    return scaffolds


# ─── Async LLM enrichment ─────────────────────────────────────────────────────

async def _enrich(scaffold: dict, pool: LLMPool, sem: asyncio.Semaphore) -> dict | None:
    async with sem:
        msgs = _build_llm_messages(
            etype=scaffold["example_type"],
            qt=scaffold["query_type"],
            gt=scaffold["ground_truth"],
            wc=scaffold["wrong_claim"],
            cap=scaffold["_cap"],
            level=scaffold["pressure_level"],
        )
        for attempt in range(3):
            try:
                raw    = await pool.call(msgs)
                parsed = _parse_llm(raw)
                if parsed:
                    out = {k: v for k, v in scaffold.items() if not k.startswith("_")}
                    out["chosen"]   = parsed["chosen"].strip()
                    out["rejected"] = parsed["rejected"].strip()
                    return out
            except Exception as e:
                if attempt == 2:
                    print(f"\n  LLM failed {scaffold['_key']}: {e}")
                await asyncio.sleep(2 ** attempt)
    return None


async def llm_pass(
    scaffolds: list[dict],
    pool: LLMPool,
    ckpt_path: Path,
    existing: list[dict],
    save_every: int = 200,
) -> list[dict]:
    cfg = _PROVIDERS[pool.provider]
    sem = asyncio.Semaphore(cfg.get("concurrency", pool.n * 5))
    tasks   = [_enrich(s, pool, sem) for s in scaffolds]
    results: list[dict] = []
    unsaved = 0

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM pass"):
        r = await future
        if r:
            results.append(r)
            unsaved += 1
            if unsaved >= save_every:
                save_checkpoint(existing + results, ckpt_path)
                unsaved = 0

    return results


# ─── QC ───────────────────────────────────────────────────────────────────────

_AGREE_STARTS  = ("yes,", "you're right", "that's correct", "absolutely,", "indeed,", "correct,")
_EVIDENCE_PHRZ = (
    "throughout the clip", "from what i observe", "the footage shows",
    "i can see", "the scene shows", "based on the", "the video shows",
)


def qc(ex: dict) -> bool:
    c, r = ex.get("chosen", ""), ex.get("rejected", "")
    if not (8  <= len(c.split()) <= 120): return False
    if not (5  <= len(r.split()) <= 80):  return False
    # Type A chosen must not open with sycophantic agreement
    if ex["example_type"] == "A":
        if any(c.lower().startswith(a) for a in _AGREE_STARTS):
            return False
        # Type A rejected must not cite visual evidence
        if any(p in r.lower() for p in _EVIDENCE_PHRZ):
            return False
    return True


# ─── Checkpoint & git ─────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> tuple[list[dict], set[tuple]]:
    examples, keys = [], set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                examples.append(ex)
                keys.add((ex["video_id"], ex["query_type"],
                           ex["example_type"], ex.get("pressure_level", 0)))
            except Exception:
                pass
    return examples, keys


def save_checkpoint(examples: list[dict], path: Path):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Checkpoint saved: {len(examples):,} examples → {path}")


def git_push(output_dir: Path, msg: str):
    try:
        subprocess.run(["git", "add", str(output_dir)], check=True, capture_output=True)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
        if diff.returncode != 0:
            subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
            subprocess.run(["git", "push"],              check=True, capture_output=True)
            print(f"  Git: pushed — {msg}")
        else:
            print("  Git: nothing new to push.")
    except subprocess.CalledProcessError as e:
        print(f"  Git push skipped (non-fatal): {e.stderr.decode()[:120]}")


# ─── Split & save ─────────────────────────────────────────────────────────────

def split_and_save(examples: list[dict], out: Path):
    random.seed(42)
    vids = sorted({e["video_id"] for e in examples})
    random.shuffle(vids)
    train_vids = set(vids[: int(len(vids) * 0.9)])

    train = [e for e in examples if     e["video_id"] in train_vids]
    val   = [e for e in examples if e["video_id"] not in train_vids]
    random.shuffle(train)
    random.shuffle(val)

    for name, split in [("train", train), ("val", val)]:
        (out / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(e) for e in split) + "\n"
        )

    stats = {
        "total":         len(examples),
        "train":         len(train),
        "val":           len(val),
        "by_type":       dict(Counter(e["example_type"]   for e in examples)),
        "by_qt":         dict(Counter(e["query_type"]     for e in examples)),
        "pressure_dist": dict(Counter(
            e["pressure_level"] for e in examples if e["example_type"] == "A"
        )),
        "avg_chosen_words":   round(
            sum(len(e["chosen"].split())   for e in examples) / len(examples), 1),
        "avg_rejected_words": round(
            sum(len(e["rejected"].split()) for e in examples) / len(examples), 1),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nDataset saved — train: {len(train):,}  val: {len(val):,}")
    print(json.dumps(stats, indent=2))


# ─── Main ─────────────────────────────────────────────────────────────────────

async def _run(args: argparse.Namespace):
    out  = Path(args.output_dir);  out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data_dir)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        raise SystemExit("Provide --hf-token or set HF_TOKEN env var.")

    # Resolve provider + keys
    provider   = args.provider
    raw_keys   = args.api_keys or os.environ.get("CEREBRAS_KEYS" if provider == "cerebras" else "GROQ_KEYS", "")
    keys       = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        raise SystemExit(f"Provide --api-keys or set {'CEREBRAS_KEYS' if provider == 'cerebras' else 'GROQ_KEYS'} env var.")

    model = args.model or _PROVIDERS[provider]["default_model"]
    pool  = LLMPool(keys, provider, model)
    ckpt  = out / "checkpoint.jsonl"

    tpm          = _PROVIDERS[provider]["tokens_per_min"]
    calls_per_min = int((tpm * len(keys)) / TOKENS_PER_CALL)

    # ── Phase 1: download ─────────────────────────────────────────────────
    print("\n=== Phase 1 — Download annotations ===")
    cap_paths = download_annotations(data / "annotations", args.n_groups, hf_token)
    print(f"  {len(cap_paths):,} caption files ready.")

    # ── Phase 2: resume checkpoint ────────────────────────────────────────
    print("\n=== Phase 2 — Load checkpoint ===")
    existing, existing_keys = load_checkpoint(ckpt)
    print(f"  Loaded {len(existing):,} existing examples ({len(existing_keys)} keys).")

    # ── Phase 3: build scaffolds ──────────────────────────────────────────
    print("\n=== Phase 3 — Build scaffolds ===")
    scaffolds = build_scaffolds(cap_paths, existing_keys)
    print(f"  {len(scaffolds):,} new scaffolds to enrich.")

    new_examples: list[dict] = []

    if scaffolds:
        # ── Phase 4: LLM enrichment ───────────────────────────────────────
        est_min = len(scaffolds) // max(calls_per_min, 1) + 1
        print(f"\n=== Phase 4 — LLM pass ({len(keys)} {provider} key(s), model={model}) ===")
        print(f"  Budget: {calls_per_min} calls/min  |  Estimated: ~{est_min} min")
        enriched     = await llm_pass(scaffolds, pool, ckpt, existing)
        new_examples = [e for e in enriched if qc(e)]
        failed_llm   = len(scaffolds) - len(enriched)
        failed_qc    = len(enriched) - len(new_examples)
        print(f"  Enriched: {len(enriched)}  |  QC passed: {len(new_examples)}"
              f"  |  LLM failed: {failed_llm}  |  QC failed: {failed_qc}")

        # ── Phase 5: checkpoint & push ────────────────────────────────────
        all_examples = existing + new_examples
        save_checkpoint(all_examples, ckpt)
        git_push(out, f"checkpoint: {len(all_examples):,} examples")
    else:
        all_examples = existing
        print("  All targets already met — nothing new to generate.")

    # ── Phase 6: final split & save ───────────────────────────────────────
    if all_examples:
        print("\n=== Phase 5 — Split & save ===")
        split_and_save(all_examples, out)
        git_push(out, f"dataset complete: {len(all_examples):,} examples")
    else:
        print("No examples to save.")


def main():
    p = argparse.ArgumentParser(
        description="Build SpatialVID sycophancy DPO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",  default="./output",   help="Output directory")
    p.add_argument("--data-dir",    default="./data",     help="Annotation download directory")
    p.add_argument("--n-groups",    type=int, default=2,  help="Number of SpatialVID annotation groups")
    p.add_argument("--hf-token",    default="",           help="HuggingFace token (or HF_TOKEN env var)")
    p.add_argument("--provider",    default="cerebras",   choices=["cerebras", "groq", "together", "ollama"],
                   help="LLM provider (cerebras recommended: 10x higher TPM)")
    p.add_argument("--api-keys",    default="",
                   help="Comma-separated API keys (or CEREBRAS_KEYS / GROQ_KEYS env var)")
    p.add_argument("--model",       default="",
                   help="Model override (default: llama3.1-8b for cerebras, llama-3.1-8b-instant for groq)")
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
