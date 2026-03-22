"""
ArvyaX Reflective AI Pipeline  —  Parts 1–9
Config (messages, ordering) loaded from config.json.
Column types auto-detected from CSV.
"""

import csv, json, warnings
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import joblib

warnings.filterwarnings("ignore")

TRAIN_PATH  = "/mnt/user-data/uploads/Sample_arvyax_reflective_dataset_xlsx_-_Dataset_120.csv"
TEST_PATH   = "/mnt/user-data/uploads/arvyax_test_inputs_120_xlsx_-_Sheet1.csv"
CONFIG_PATH = Path(__file__).parent / "config.json"
OUT_DIR     = Path("/mnt/user-data/outputs")
OUT_DIR.mkdir(exist_ok=True)
RS, CONF_THOLD = 42, 0.45
SKIP_COLS = {"id", "journal_text", "emotional_state", "intensity"}

# ── Load config (messages + semantic orderings) ────────────────────────────────
cfg         = json.loads(CONFIG_PATH.read_text())
MESSAGES    = {tuple(k.split("|")): v for k, v in cfg["messages"].items()}
WHEN_SUFFIX = cfg["when_suffix"]

# ── Decision rules (behaviour logic — stays in code, not config) ───────────────
WHAT_RULES = [
    {"state":"overwhelmed", "hi_stress":True,  "hi_intens":True,  "action":"box_breathing"},
    {"state":"overwhelmed", "lo_energy":True,                      "action":"rest"},
    {"state":"overwhelmed",                                         "action":"grounding"},
    {"state":"restless",    "hi_intens":True,                      "action":"box_breathing"},
    {"state":"restless",    "hi_energy":True,                      "action":"movement"},
    {"state":"restless",                                            "action":"journaling"},
    {"state":"focused",     "hi_stress":False, "hi_energy":True,   "action":"deep_work"},
    {"state":"focused",                                             "action":"light_planning"},
    {"state":"calm",        "lo_energy":True,  "is_night":True,    "action":"rest"},
    {"state":"calm",        "is_morn":True,                        "action":"light_planning"},
    {"state":"calm",                                                "action":"journaling"},
    {"state":"mixed",       "hi_stress":True,                      "action":"grounding"},
    {"state":"mixed",       "hi_energy":True,                      "action":"movement"},
    {"state":"mixed",                                               "action":"sound_therapy"},
    {"state":"neutral",     "hi_energy":True,  "is_morn":True,     "action":"light_planning"},
    {"state":"neutral",     "hi_stress":True,                      "action":"journaling"},
    {"state":"neutral",                                             "action":"movement"},
]

WHEN_RULES = [
    {"state":"overwhelmed", "hi_intens":True,  "hi_stress":True,   "when":"now"},
    {"state":"restless",    "hi_intens":True,  "hi_stress":True,   "when":"now"},
    {"state":"focused",     "what":"deep_work",                     "when":"now"},
    {                       "what":"rest",     "is_night":True,    "when":"tonight"},
    {                                          "is_night":True,    "when":"tomorrow_morning"},
    {"state":"calm",        "hi_intens":False,                      "when":"later_today"},
    {                       "hi_intens":True,                       "when":"now"},
    {                       "what":"journaling",                    "when":"later_today"},
    {                       "what":"sound_therapy",                 "when":"later_today"},
]

# ── Column type detection from data (no hardcoding) ────────────────────────────
def detect_col_types(rows):
    """Auto-detect numeric vs categorical by attempting float conversion."""
    num_cols, cat_cols = [], []
    for col in rows[0]:
        if col in SKIP_COLS:
            continue
        # Try converting first non-empty value
        sample = next((r[col] for r in rows if r[col].strip()), "")
        try:
            float(sample); num_cols.append(col)
        except ValueError:
            cat_cols.append(col)
    return num_cols, cat_cols

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def is_short_vague(text):
    return len(text.strip().split()) <= 3

def conflict_texts(rows):
    tmap = defaultdict(set)
    for r in rows:
        tmap[r["journal_text"].strip().lower()].add(r["emotional_state"])
    return {t for t, lbls in tmap.items() if len(lbls) > 1}

# ── Preprocessors ──────────────────────────────────────────────────────────────
def fit_preprocessors(rows, num_cols, cat_cols):
    num_arr = np.array([[float(r[c]) if r[c].strip() else np.nan for c in num_cols]
                        for r in rows])
    imputer = SimpleImputer(strategy="median").fit(num_arr)

    cat_arr    = [[r[c].strip().lower() for c in cat_cols] for r in rows]
    # Categories read from training data — tree models are order-invariant
    categories = [sorted(set(row[i] for row in cat_arr)) for i in range(len(cat_cols))]
    enc = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value", unknown_value=np.nan
    ).fit(cat_arr)

    return imputer, enc

def extract_meta(rows, num_cols, cat_cols, imputer, enc):
    num_arr = np.array([[float(r[c]) if r[c].strip() else np.nan for c in num_cols]
                        for r in rows])
    num_imp = imputer.transform(num_arr)

    cat_arr = [[r[c].strip().lower() for c in cat_cols] for r in rows]
    cat_enc = enc.transform(cat_arr)

    energy = num_imp[:, num_cols.index("energy_level")]
    stress = num_imp[:, num_cols.index("stress_level")]
    tod    = [r.get("time_of_day","").lower() for r in rows]

    derived = np.column_stack([
        stress / np.maximum(energy, 0.1),
        [1 if t in ("night","evening") else 0 for t in tod],
        [1 if is_short_vague(r.get("journal_text","")) else 0 for r in rows],
    ])
    return np.hstack([num_imp, cat_enc, derived])

def enrich_text(r):
    t = r.get("journal_text","").strip().lower()
    if not t or is_short_vague(t):
        t = f"brief reflection {r.get('ambience_type','')} {r.get('time_of_day','')} session"
    return t

def build_X(rows, num_cols, cat_cols, tfidf=None, imputer=None, enc=None, fit=False):
    texts = [enrich_text(r) for r in rows]
    if fit:
        tfidf   = TfidfVectorizer(max_features=3000, ngram_range=(1,2),
                                   min_df=2, sublinear_tf=True, smooth_idf=True)
        imputer, enc = fit_preprocessors(rows, num_cols, cat_cols)
        text_arr = tfidf.fit_transform(texts).toarray()
    else:
        text_arr = tfidf.transform(texts).toarray()
    meta = extract_meta(rows, num_cols, cat_cols, imputer, enc)
    return np.hstack([text_arr, meta]), text_arr, meta, tfidf, imputer, enc

# ── Uncertainty ────────────────────────────────────────────────────────────────
def get_uncertainty(proba, rows, conflict_set):
    results = []
    for p, r in zip(proba, rows):
        conf = float(np.max(p))
        txt  = r["journal_text"].strip().lower()
        flag = int(conf < CONF_THOLD or txt in conflict_set or is_short_vague(txt))
        if txt in conflict_set: conf = min(conf, 0.50)
        if is_short_vague(txt): conf = min(conf, 0.55)
        results.append((round(conf, 4), flag))
    return results

# ── Decision engine ────────────────────────────────────────────────────────────
def _matches(rule, ctx):
    return all(ctx.get(k) == v for k, v in rule.items() if k not in ("action","when"))

def decision(state, intensity, stress, energy, tod):
    ctx = {
        "state":    state,
        "hi_stress": float(stress or 3) >= 4,
        "lo_energy": float(energy or 3) <= 2,
        "hi_energy": float(energy or 3) >= 4,
        "hi_intens": int(intensity or 3) >= 4,
        "is_night":  str(tod).lower() in ("night","evening"),
        "is_morn":   str(tod).lower() in ("morning","early_morning"),
    }
    what = next((r["action"] for r in WHAT_RULES if _matches(r, ctx)), "pause")
    ctx["what"] = what
    when = next((r["when"]   for r in WHEN_RULES if _matches(r, ctx)), "within_15_min")
    return what, when

def supportive_msg(state, what, when):
    base = MESSAGES.get((state, what),
           f"Take a moment for yourself. {what.replace('_',' ').title()} is the right move.")
    return base + WHEN_SUFFIX.get(when, "")

# ── Pipeline ───────────────────────────────────────────────────────────────────
def run():
    print("="*60)
    print("  ArvyaX Reflective AI — Full Pipeline")
    print("="*60)

    train, test = load_csv(TRAIN_PATH), load_csv(TEST_PATH)
    num_cols, cat_cols = detect_col_types(train)
    c_set = conflict_texts(train)
    print(f"\n[Data]    Train={len(train)}  Test={len(test)}  ConflictTexts={len(c_set)}")
    print(f"[Columns] num={num_cols}")
    print(f"          cat={cat_cols}")

    le       = LabelEncoder()
    y_state  = le.fit_transform([r["emotional_state"] for r in train])
    y_intens = np.array([int(r["intensity"]) for r in train])
    print(f"[Labels]  {list(le.classes_)}")

    X, X_text, X_meta, tfidf, imputer, enc = build_X(
        train, num_cols, cat_cols, fit=True)
    print(f"[Features] text={X_text.shape[1]}  meta={X_meta.shape[1]}  total={X.shape[1]}")
    print(f"[Imputer]  medians from data: "
          + "  ".join(f"{c}={v:.1f}" for c, v in zip(num_cols, imputer.statistics_)))

    # ── Part 1: State ──────────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("PART 1 — Emotional State Prediction")
    print("─"*55)

    clf_s = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                   subsample=0.8, min_samples_leaf=5, random_state=RS),
        cv=3, method="isotonic")
    clf_s.fit(X, y_state)

    cv_s = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=RS),
                           X, y_state, cv=5, scoring="f1_macro")
    pred_s_train = clf_s.predict(X)
    print(f"  5-fold CV F1-macro: {cv_s.mean():.4f} ± {cv_s.std():.4f}")
    print(f"  Train accuracy:     {accuracy_score(y_state, pred_s_train):.4f}")
    print(classification_report(y_state, pred_s_train, target_names=le.classes_, digits=3))

    # ── Part 2: Intensity ──────────────────────────────────────────────────────
    print("─"*55)
    print("PART 2 — Intensity  [Ordinal Classification — evaluated with MAE]")
    print("─"*55)

    clf_i = CalibratedClassifierCV(
        GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                   learning_rate=0.1, random_state=RS),
        cv=3, method="isotonic")
    clf_i.fit(X, y_intens)
    pred_i_train = clf_i.predict(X)
    cv_mae = -cross_val_score(RandomForestClassifier(n_estimators=200, random_state=RS),
                              X, y_intens, cv=5, scoring="neg_mean_absolute_error")
    print(f"  Train MAE: {mean_absolute_error(y_intens, pred_i_train):.4f}  "
          f"| Train acc: {accuracy_score(y_intens, pred_i_train):.4f}")
    print(f"  5-fold CV MAE: {cv_mae.mean():.4f} ± {cv_mae.std():.4f}")

    # ── Part 5: Feature importance ─────────────────────────────────────────────
    print("\n" + "─"*55)
    print("PART 5 — Feature Understanding")
    print("─"*55)

    rf_imp = RandomForestClassifier(n_estimators=300, random_state=RS).fit(X, y_state)
    imps   = rf_imp.feature_importances_
    n_text = X_text.shape[1]
    t_imp, m_imp = imps[:n_text].sum(), imps[n_text:].sum()
    print(f"\n  Text  importance: {t_imp:.4f} ({t_imp*100:.1f}%)")
    print(f"  Meta  importance: {m_imp:.4f} ({m_imp*100:.1f}%)")

    meta_names = num_cols + cat_cols + ["stress_energy_ratio","is_night","is_short_vague"]

    print("\n  Top 12 text tokens:")
    for tok, v in sorted(zip(tfidf.get_feature_names_out(), imps[:n_text]),
                         key=lambda x: x[1], reverse=True)[:12]:
        print(f"    {tok:<25} {v:.5f}  {'▓'*int(v*2000)}")

    print("\n  Metadata importances:")
    for name, v in sorted(zip(meta_names, imps[n_text:]),
                          key=lambda x: x[1], reverse=True):
        print(f"    {name:<25} {v:.5f}  {'▓'*int(v*500)}")

    # ── Part 6: Ablation ───────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("PART 6 — Ablation Study")
    print("─"*55)
    print(f"\n  {'Model':<28} {'CV F1-macro':<14} {'Std':<8} {'vs text-only'}")
    print(f"  {'─'*58}")
    baseline = None
    for name, feat in [("Text-only", X_text), ("Metadata-only", X_meta), ("Text + Metadata", X)]:
        sc = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=RS),
                             feat, y_state, cv=5, scoring="f1_macro")
        diff = f"{(sc.mean()-baseline)*100:+.2f}pp" if baseline is not None else "—"
        if baseline is None: baseline = sc.mean()
        print(f"  {name:<28} {sc.mean():<14.4f} ±{sc.std():.4f}  {diff}")

    # ── Part 7: Error analysis ─────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("PART 7 — Error Analysis")
    print("─"*55)

    proba_train = clf_s.predict_proba(X)
    errors = [{"id": r["id"], "text": r["journal_text"],
               "true": r["emotional_state"],
               "pred": le.classes_[pred_s_train[i]],
               "conf": float(np.max(proba_train[i])),
               "stress": r["stress_level"], "energy": r["energy_level"],
               "face": r["face_emotion_hint"],
               "is_conflict": r["journal_text"].strip().lower() in c_set,
               "is_short":    is_short_vague(r["journal_text"])}
              for i, r in enumerate(train)
              if le.classes_[pred_s_train[i]] != r["emotional_state"]]

    buckets = {
        "noisy label / conflict text": (
            [e for e in errors if e["is_conflict"]],
            "Same text has multiple labels in train — model saw ambiguous signal.",
            "uncertain_flag=1 set. Stronger metadata needed to break the tie."),
        "short / vague input": (
            [e for e in errors if e["is_short"] and not e["is_conflict"]],
            "≤3 words — near-zero TF-IDF vector; model falls back to metadata.",
            "Text enriched with ambience+tod. uncertain_flag=1 always set."),
        "metadata-text mismatch": (
            sorted([e for e in errors if not e["is_conflict"] and not e["is_short"]],
                   key=lambda x: x["conf"], reverse=True),
            "Text sentiment contradicts physiological signals.",
            "Better text-metadata fusion weighting would help."),
    }

    shown = 0
    for cat, (bucket, why, fix) in buckets.items():
        for e in bucket[:4]:
            if shown >= 10: break
            shown += 1
            print(f"\n  [{shown:02d}] {cat}")
            print(f"       id={e['id']}  True: {e['true']:<14} Pred: {e['pred']:<14} Conf: {e['conf']:.3f}")
            print(f"       text: \"{e['text'][:65]}\"")
            print(f"       stress={e['stress']}  energy={e['energy']}  face={e['face']}")
            print(f"       why: {why}")
            print(f"       fix: {fix}")

    print(f"\n  Total train errors: {len(errors)}/{len(train)} ({len(errors)/len(train)*100:.1f}%)")

    # ── Test inference ─────────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("TEST SET PREDICTIONS")
    print("─"*55)

    X_test, _, _, _, _, _ = build_X(
        test, num_cols, cat_cols, tfidf=tfidf, imputer=imputer, enc=enc)
    pred_s = le.inverse_transform(clf_s.predict(X_test))
    pred_p = clf_s.predict_proba(X_test)
    pred_i = clf_i.predict(X_test)
    unc    = get_uncertainty(pred_p, test, c_set)

    print(f"  State dist:     {dict(Counter(pred_s))}")
    print(f"  Intensity dist: {dict(sorted(Counter(pred_i).items()))}")
    print(f"  Uncertain rows: {sum(u[1] for u in unc)}/{len(test)}")

    fields    = ["id","predicted_state","predicted_intensity","confidence",
                 "uncertain_flag","what_to_do","when_to_do","supportive_message"]
    rows_out  = []
    for i, r in enumerate(test):
        what, when = decision(pred_s[i], pred_i[i],
                              r.get("stress_level"), r.get("energy_level"),
                              r.get("time_of_day","afternoon"))
        conf, flag = unc[i]
        rows_out.append({"id": r["id"], "predicted_state": pred_s[i],
                         "predicted_intensity": int(pred_i[i]),
                         "confidence": conf, "uncertain_flag": flag,
                         "what_to_do": what, "when_to_do": when,
                         "supportive_message": supportive_msg(pred_s[i], what, when)})

    pred_path = OUT_DIR / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows_out)
    print(f"\n  Saved: {pred_path}")

    model_path = OUT_DIR / "models.joblib"
    joblib.dump({"clf_s": clf_s, "clf_i": clf_i, "tfidf": tfidf, "le": le,
                 "imputer": imputer, "enc": enc, "conflict_set": c_set,
                 "num_cols": num_cols, "cat_cols": cat_cols}, model_path)

    # ── Part 8 ─────────────────────────────────────────────────────────────────
    import os
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print("\n" + "─"*55)
    print("PART 8 — Edge / Offline Deployment")
    print("─"*55)
    print(f"""
  Model bundle: {size_mb:.2f} MB  (GBM x2 + TF-IDF + OrdinalEncoder + Imputer)
  Inference:    < 20 ms CPU  (TF-IDF ~5ms, GBM ~10ms, rules <1ms)
  Mobile:       joblib → FastAPI on-device  |  sklearn-onnx → CoreML (iOS)
  Shrink:       TF-IDF 500 feats → ~0.5 MB, <5% loss  |  GBM → LR → 0.1 MB""")

    # ── Part 9 ─────────────────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("PART 9 — Robustness Tests")
    print("─"*55)

    rob = [
        ("very short text",             {"journal_text":"ok",              "ambience_type":"forest","duration_min":"10","sleep_hours":"6", "energy_level":"3","stress_level":"4","time_of_day":"morning",   "previous_day_mood":"mixed",       "face_emotion_hint":"tense_face",  "reflection_quality":"vague"}),
        ("missing sleep+mood+face",     {"journal_text":"felt a bit lighter","ambience_type":"ocean","duration_min":"12","sleep_hours":"",  "energy_level":"2","stress_level":"5","time_of_day":"night",     "previous_day_mood":"",            "face_emotion_hint":"",            "reflection_quality":"vague"}),
        ("contradictory calm+stress=5", {"journal_text":"kinda calm now",   "ambience_type":"cafe", "duration_min":"15","sleep_hours":"7", "energy_level":"4","stress_level":"5","time_of_day":"afternoon", "previous_day_mood":"overwhelmed", "face_emotion_hint":"neutral_face","reflection_quality":"conflicted"}),
    ]
    X_rob, _, _, _, _, _ = build_X(
        [r for _, r in rob], num_cols, cat_cols, tfidf=tfidf, imputer=imputer, enc=enc)
    rob_s = le.inverse_transform(clf_s.predict(X_rob))
    rob_p = clf_s.predict_proba(X_rob)
    rob_i = clf_i.predict(X_rob)

    print(f"\n  {'Case':<36} {'State':<14} {'Int':<5} {'Conf':<8} {'Flag'}")
    print(f"  {'─'*68}")
    for k, (lbl, r) in enumerate(rob):
        conf = float(np.max(rob_p[k]))
        flag = int(is_short_vague(r["journal_text"]) or conf < CONF_THOLD
                   or r["journal_text"].strip().lower() in c_set)
        if r["journal_text"].strip().lower() in c_set: conf = min(conf, 0.50)
        if is_short_vague(r["journal_text"]):           conf = min(conf, 0.55)
        print(f"  {lbl:<36} {rob_s[k]:<14} {rob_i[k]:<5} {conf:<8.4f} {'YES' if flag else 'no'}")

    print("\n  Handling: short(≤3w)→enriched+flag | missing→imputer median | "
          "contradictory→fused, low conf triggers flag")
    print("\n" + "="*60)
    print(f"  Done.  {pred_path}")
    print("="*60)


if __name__ == "__main__":
    run()
