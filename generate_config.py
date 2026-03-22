"""
generate_config.py
Generates config.json from two sources:
  1. CSV data  — unique category values per column (auto-detected)
  2. This file — semantic ordering + messages (domain knowledge, defined once here)

Run whenever the CSV schema changes. Do not hand-edit config.json directly.
"""

import csv, json
from pathlib import Path

TRAIN_PATH  = "/mnt/user-data/uploads/Sample_arvyax_reflective_dataset_xlsx_-_Dataset_120.csv"
OUT_PATH    = Path(__file__).parent / "config.json"
SKIP_COLS   = {"id", "journal_text", "emotional_state", "intensity"}

# ── Domain knowledge: the ORDER within each categorical column ─────────────────
# Values must match exactly what appears in the CSV (lowercase, stripped).
# Order encodes semantic direction — cannot be derived from data.
#   face/mood: calm-aroused axis
#   time_of_day: temporal progression
#   reflection_quality: signal clarity
#   ambience_type: no inherent order → alphabetical is fine
SEMANTIC_ORDER = {
    "face_emotion_hint": ["calm_face","happy_face","neutral_face","tense_face","tired_face","none",""],
    "time_of_day":       ["early_morning","morning","afternoon","evening","night"],
    "previous_day_mood": ["calm","focused","neutral","mixed","restless","overwhelmed",""],
    "reflection_quality":["clear","conflicted","vague"],
}
# Columns not listed above get values sorted alphabetically (no semantic axis).

# ── Product copy: messages and timing suffixes ─────────────────────────────────
# Key format: "state|action" — pipe-separated to stay valid JSON keys.
MESSAGES = {
    "overwhelmed|box_breathing": "You're carrying a lot. Let's slow your nervous system down with box breathing first.",
    "overwhelmed|rest":          "Your body and mind need recovery. Rest without guilt — it's productive.",
    "overwhelmed|grounding":     "Things feel heavy. A quick grounding exercise can bring you back to the present.",
    "restless|box_breathing":    "There's a buzzing energy that needs settling. A short breath reset will help.",
    "restless|movement":         "Channel that restless energy into movement before it spirals.",
    "restless|journaling":       "Writing it out can help you find the thread under the restlessness.",
    "focused|deep_work":         "You're in a good headspace. Use this window — do your most important task now.",
    "focused|light_planning":    "Your clarity is an asset. Sketch your priorities while thinking is sharp.",
    "calm|rest":                 "You've found stillness. Honour it — rest is the move right now.",
    "calm|journaling":           "This calm is a good moment to reflect and capture what's working.",
    "calm|light_planning":       "A calm mind is a planning mind. Map your next steps gently.",
    "mixed|grounding":           "Your signals are mixed. Start with grounding to find your centre.",
    "mixed|sound_therapy":       "Ambient sound can help you find a neutral baseline when things feel blended.",
    "mixed|movement":            "Physical movement can help untangle the mental mix.",
    "neutral|light_planning":    "Steady state is a great launchpad. A little structure sets up your day.",
    "neutral|journaling":        "Neutral can mean unexpressed. Journaling may surface what's sitting below.",
    "neutral|movement":          "A little movement can nudge you from neutral toward energy.",
}

WHEN_SUFFIX = {
    "now":              " Start right now.",
    "within_15_min":    " Try this in the next 15 minutes.",
    "later_today":      " Aim for this later today.",
    "tonight":          " Save this for tonight.",
    "tomorrow_morning": " Plan this for tomorrow morning.",
}

# ── Build cat_cols_ordered from data + semantic ordering ───────────────────────
def build_cat_ordered(path):
    rows = list(csv.DictReader(open(path)))
    result = {}
    for col in rows[0]:
        if col in SKIP_COLS:
            continue
        # detect categorical: first non-empty value fails float()
        sample = next((r[col] for r in rows if r[col].strip()), "")
        try:
            float(sample)
            continue                          # numeric — skip
        except ValueError:
            pass

        # collect unique values from data (lowercase, stripped)
        data_vals = {r[col].strip().lower() for r in rows}

        if col in SEMANTIC_ORDER:
            ordered = SEMANTIC_ORDER[col]
            # Warn if data has values not covered by semantic ordering
            missing = data_vals - set(ordered)
            if missing:
                print(f"  WARNING: {col} has values in data not in SEMANTIC_ORDER: {missing}")
                ordered = ordered + sorted(missing)   # append extras at end
        else:
            ordered = sorted(data_vals)               # alphabetical fallback

        result[col] = ordered
    return result


def main():
    print(f"Reading: {TRAIN_PATH}")
    cat_cols_ordered = build_cat_ordered(TRAIN_PATH)

    print("Detected categorical columns:")
    for col, vals in cat_cols_ordered.items():
        print(f"  {col}: {vals}")

    config = {
        "_note": "Generated by generate_config.py — do not edit manually.",
        "_note_ordering": (
            "Semantic order for face/mood/tod/quality encodes domain knowledge "
            "(calm→aroused, temporal progression). Cannot be derived from data."
        ),
        "cat_cols_ordered": cat_cols_ordered,
        "messages":         MESSAGES,
        "when_suffix":      WHEN_SUFFIX,
    }

    OUT_PATH.write_text(json.dumps(config, indent=2))
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
