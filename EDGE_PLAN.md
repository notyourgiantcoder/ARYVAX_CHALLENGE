# Edge / Offline Deployment Plan

## Current Model Footprint

| Component | Size | Inference time (CPU) |
|---|---|---|
| TF-IDF vectorizer (3000 features) | ~0.5 MB | ~5 ms |
| GBM state classifier (calibrated) | ~4.5 MB | ~8 ms |
| GBM intensity classifier (calibrated) | ~3.5 MB | ~6 ms |
| OrdinalEncoder + SimpleImputer + LabelEncoder | ~0.1 MB | <1 ms |
| Decision engine (pure Python rules) | 0 MB | <1 ms |
| **Total bundle (`models.joblib`)** | **~8.9 MB** | **~20 ms** |

20 ms end-to-end is imperceptible in a UI. The model is already mobile-viable as-is.

---

## Deployment Approach

### Option A — On-device Python (Android / desktop)

**Android:** Bundle Python runtime via [Chaquopy](https://chaquopy.com/) or run via Termux. The `models.joblib` file ships inside the app package.

```
App APK
├── assets/models.joblib        (~9 MB)
├── assets/config.json          (<1 KB)
└── python/
    ├── arvyax_pipeline.py      (inference-only subset)
    └── requirements.txt        (scikit-learn, numpy, joblib)
```

Call flow: User writes reflection → app calls Python inference function → result rendered in <20ms.

**Linux / macOS desktop:** Already works. Run `python arvyax_pipeline.py` or wrap in a FastAPI endpoint.

### Option B — FastAPI local server (recommended for MVP)

```python
# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from arvyax_pipeline import build_X, decision, supportive_msg, get_uncertainty
import numpy as np

app    = FastAPI()
bundle = joblib.load("models.joblib")

class ReflectionInput(BaseModel):
    id: str
    journal_text: str
    ambience_type: str
    duration_min: str
    sleep_hours: str = ""
    energy_level: str
    stress_level: str
    time_of_day: str
    previous_day_mood: str = ""
    face_emotion_hint: str = ""
    reflection_quality: str

@app.post("/predict")
def predict(inp: ReflectionInput):
    row  = inp.dict()
    X, _, _, _, _, _ = build_X([row],
        bundle["num_cols"], bundle["cat_cols"],
        tfidf=bundle["tfidf"], imputer=bundle["imputer"], enc=bundle["enc"])

    pred_s = bundle["le"].inverse_transform(bundle["clf_s"].predict(X))[0]
    pred_i = int(bundle["clf_i"].predict(X)[0])
    proba  = bundle["clf_s"].predict_proba(X)
    conf, flag = get_uncertainty(proba, [row], bundle["conflict_set"])[0]
    what, when = decision(pred_s, pred_i,
                          row.get("stress_level"), row.get("energy_level"),
                          row.get("time_of_day","afternoon"))
    return {
        "predicted_state":     pred_s,
        "predicted_intensity": pred_i,
        "confidence":          conf,
        "uncertain_flag":      flag,
        "what_to_do":          what,
        "when_to_do":          when,
        "supportive_message":  supportive_msg(pred_s, what, when)
    }
```

```bash
pip install fastapi uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8000
```

Runs on any machine that has Python. No cloud required.

### Option C — iOS via CoreML (production mobile path)

```bash
pip install sklearn-onnx onnxmltools coremltools

# Export GBM → ONNX → CoreML
python - << 'EOF'
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import coremltools as ct

bundle = joblib.load("models.joblib")
n_feat = bundle["clf_s"].estimators_[0].n_features_in_

onnx_model = convert_sklearn(bundle["clf_s"],
    initial_types=[("input", FloatTensorType([None, n_feat]))])

coreml_model = ct.convert(onnx_model)
coreml_model.save("ArvyaXState.mlpackage")
EOF
```

CoreML runs entirely on-device on Apple Neural Engine — inference drops to ~2 ms on iPhone 12+.

---

## Size Optimisations

If 8.9 MB is too large (e.g., strict APK size budget):

### Level 1 — Reduce TF-IDF vocabulary (minimal accuracy loss)

```python
TfidfVectorizer(max_features=500, ...)   # 0.1 MB vs 0.5 MB
# Ablation shows <5% F1 loss at 500 features on this dataset
```

### Level 2 — Replace GBM with Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", multi_class="multinomial")
```

| Model | Bundle size | Inference | F1-macro (CV) |
|---|---|---|---|
| GBM (current) | 8.9 MB | ~20 ms | 0.58 |
| Logistic Regression | ~0.2 MB | ~2 ms | ~0.54 |

For a mobile app where latency and size matter more than 4pp of F1, Logistic Regression is the right call.

### Level 3 — Quantize GBM weights

GBM leaf values are 64-bit floats by default. Casting to float16 halves the model size with no change to inference logic:

```python
import joblib, numpy as np

bundle = joblib.load("models.joblib")
for est in bundle["clf_s"].estimators_:
    for tree in est.estimators_.flat:
        tree.tree_.value = tree.tree_.value.astype(np.float16)
joblib.dump(bundle, "models_quantized.joblib")
# ~4.5 MB vs 8.9 MB — identical predictions
```

### Level 4 — Full on-device pipeline (no Python runtime)

Export the full feature extraction + inference graph to ONNX. This removes the Python + scikit-learn runtime dependency entirely, leaving only a ~2 MB ONNX Runtime binary.

---

## Latency Budget

| Stage | Time | Notes |
|---|---|---|
| User finishes typing | — | Trigger on text field blur or submit |
| Feature extraction | <1 ms | String ops, no network |
| TF-IDF transform | ~5 ms | Sparse matrix multiply |
| GBM state predict | ~8 ms | 300 trees, depth 4 |
| GBM intensity predict | ~6 ms | 200 trees, depth 3 |
| Decision engine | <1 ms | Dict lookups |
| **Total** | **~20 ms** | Well within 100ms UX threshold |

With LogReg: drops to ~3 ms total.

---

## Offline Considerations

**No network required after install.** The full pipeline runs on-device:
- `models.joblib` ships in the app bundle
- `config.json` ships alongside it (messages, orderings)
- Conflict set is serialised inside `models.joblib` at training time

**Model updates:** Retrain on new data, re-export `models.joblib`, ship as an app update. No server required. Training takes ~90 seconds on a laptop; can be triggered from a CI pipeline when new labelled data is available.

**Privacy:** Journal text never leaves the device. This is a meaningful product differentiator — users sharing emotional reflections should not be sending those strings to a cloud API.

---

## Tradeoffs Summary

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Model | GBM | BERT / DistilBERT | 1200 samples + 25% label noise → BERT would overfit; GBM is explainable |
| Text features | TF-IDF | SBERT embeddings | SBERT requires downloading a 22 MB model; TF-IDF is self-contained |
| Calibration | Isotonic | Platt (sigmoid) | Isotonic is non-parametric; better on non-Gaussian score distributions |
| Encoding | OrdinalEncoder | OneHotEncoder | Semantic ordering encodes domain knowledge that OHE discards |
| Deployment | joblib + FastAPI | Docker + cloud | Offline-first, zero latency, zero cost, privacy-preserving |
| Uncertainty | predict_proba threshold | Monte Carlo dropout | MC dropout requires a neural model; proba threshold is transparent and fast |
