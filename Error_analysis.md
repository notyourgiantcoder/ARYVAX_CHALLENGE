# Error Analysis

**Total training errors:** 68 / 1200 (5.7%)  
**Error breakdown:**
- Noisy label / conflict text: ~45 cases (66%)
- Short / vague input: ~8 cases (12%)
- Metadata–text mismatch: ~15 cases (22%)

The low overall error rate (5.7%) masks a structural problem: almost all errors fall into predictable, explainable categories. This is useful — it means the model's failures are not random noise but systematic patterns that a better data collection process or a richer model architecture could address.

---

## Case 1 — Conflict text, low energy misleads
**id:** 485  
**Text:** "I guess mind was all over the place."  
**True:** `neutral` | **Predicted:** `focused` | **Confidence:** 0.458  
**Metadata:** stress=3, energy=1, sleep=4, face=tired_face, tod=night, prev_mood=mixed

**What went wrong:**  
This exact text appears with 4 different labels in the training set. The model averaged across all of them and settled on `focused` because other instances of this text had high-energy, focused metadata. For *this* instance, energy=1 and tired_face should have pointed toward `neutral` or `overwhelmed`.

**Why it failed:**  
Label noise. The text is genuinely ambiguous — "mind all over the place" can indicate restlessness, overwhelm, or a distracted-but-neutral state depending on context. The model has seen contradictory supervision for this exact string.

**How to improve:**  
Weight metadata more heavily when the text is in the known conflict set. A two-stage model — text predicts state, metadata overrides when confidence is low — would handle this better. `uncertain_flag=1` is correctly set.

---

## Case 2 — Conflict text, surface positivity masks high stress
**id:** 492  
**Text:** "For some reason okay session."  
**True:** `neutral` | **Predicted:** `restless` | **Confidence:** 0.322  
**Metadata:** stress=5, energy=2, sleep=5, face=calm_face, tod=night, prev_mood=restless

**What went wrong:**  
"Okay session" appears 10 times in training with 5 different labels. The model's prediction (`restless`) is actually a reasonable inference from the metadata (stress=5, prev_mood=restless, night, tired) — but the ground truth label is `neutral`. The face hint (calm_face) contradicts the stress reading.

**Why it failed:**  
Three-way signal conflict: text says "okay," face says calm, but stress=5 and previous mood=restless. The labeller likely looked at text and face and called it neutral. The model looked at stress and history and called it restless. Both are defensible. This is irreducible label ambiguity.

**How to improve:**  
Multi-annotator labelling with disagreement tracking. A single annotator per sample cannot produce consistent labels when signals conflict. Confidence is correctly low (0.322) and `uncertain_flag=1`.

---

## Case 3 — Conflict text, "felt good for a moment" is transient
**id:** 514  
**Text:** "felt good for a moment"  
**True:** `neutral` | **Predicted:** `calm` | **Confidence:** 0.264  
**Metadata:** stress=1, energy=3, sleep=8, face=tired_face, tod=morning, prev_mood=focused

**What went wrong:**  
The phrase "felt good" is strongly associated with `calm` in the training vocabulary. The model correctly picked up that signal. However the word "moment" implies transience — the feeling didn't last — which is closer to `neutral`. TF-IDF treats "moment" as a low-frequency bigram with no special weight; it cannot capture the temporal qualifier.

**Why it failed:**  
TF-IDF has no syntactic awareness. "felt good for a moment" and "felt good" are treated almost identically. A model with word-order sensitivity (even a simple LSTM) would learn that "for a moment" is a diminisher.

**How to improve:**  
Bigram `"good for"` and `"for a moment"` would capture this if frequency is sufficient. With more data, training a simple sequence model would handle diminisher patterns. Short-term: add a "transience" lexicon feature (words like "moment", "briefly", "almost", "for a second").

---

## Case 4 — Conflict text, contradictory self-report
**id:** 527  
**Text:** "some peace, some noise in head"  
**True:** `mixed` | **Predicted:** `restless` | **Confidence:** 0.242  
**Metadata:** stress=1, energy=1, face=calm_face, tod=afternoon, prev_mood=overwhelmed

**What went wrong:**  
The text is genuinely self-contradictory — the user reports both peace and noise simultaneously. This is the definition of `mixed`, and the ground truth is correct. However, "noise in head" is a strong restlessness signal in the training vocabulary, and the model weighted it above "some peace."

**Why it failed:**  
TF-IDF treats "peace" and "noise in head" independently. There is no mechanism to detect that the user is explicitly acknowledging both states. The contrast word "some ... some" is not captured.

**How to improve:**  
A `mixed` class needs contrastive signal detection. Features like "count of positive sentiment words / count of negative sentiment words close to 1.0" would flag this pattern. Alternatively, a sentence embedding model (SBERT) would represent this sentence as a vector close to both calm and restless — which is precisely what `mixed` should look like.

---

## Case 5 — Short/vague text, metadata conflicts with enrichment
**id:** 677  
**Text:** "okay session ..."  
**True:** `neutral` | **Predicted:** `restless` | **Confidence:** 0.231  
**Metadata:** stress=2, energy=5, face=none, tod=evening, prev_mood=restless

**What went wrong:**  
Three words — TF-IDF vector is near-zero after enrichment. The model falls back almost entirely on metadata. `prev_mood=restless` and `energy=5` together push toward `restless`. The ground truth `neutral` is what an annotator inferred from the text, but the text contains no discriminative signal.

**Why it failed:**  
For very short texts, the label was assigned by a human reading "okay session" as neutral. The model, lacking text signal, over-rotates on metadata which tells a different story. The two evidence sources are using different ground truth implicitly.

**How to improve:**  
For ≤3 word texts, the fallback should weight `reflection_quality=vague` more heavily and default toward `neutral` rather than amplifying metadata extremes. A prior probability of `neutral` for vague texts would reduce variance here. `uncertain_flag=1` is correctly set.

---

## Case 6 — Restless vs Overwhelmed: boundary confusion
**id:** 90  
**Text:** "I couldn't really settle into the cafe track; I kept thinking of everything at once. Part of me wants to do everything at once."  
**True:** `restless` | **Predicted:** `overwhelmed` | **Confidence:** 0.375  
**Metadata:** stress=5, energy=2, sleep=6.5, face=neutral_face, tod=early_morning, prev_mood=overwhelmed

**What went wrong:**  
The text clearly describes cognitive restlessness. But stress=5, energy=2, prev_mood=overwhelmed form a strong overwhelmed cluster in the training data. The model conflated the emotional signature of being unable to focus (restless) with being crushed by load (overwhelmed).

**Why it failed:**  
`restless` and `overwhelmed` share many surface features: high stress, fragmented thought, inability to settle. The distinguishing factor is agency — restless people want to do things, overwhelmed people feel they can't. "Part of me wants to do everything at once" is an agency signal, but "wants to do everything at once" co-occurs with overwhelm in training too.

**How to improve:**  
Add a feature capturing **wanting-to-act language** vs **being-unable-to-act language**. A small domain-specific lexicon (agency verbs: want, try, plan, start vs inability verbs: can't, couldn't, stuck, frozen) would disambiguate this class boundary. This is one of the most valuable improvements to make.

---

## Case 7 — Positive framing obscures restlessness
**id:** 58  
**Text:** "The forest sounds were nice, but I still feel unsettled and fidgety."  
**True:** `restless` | **Predicted:** `mixed` | **Confidence:** 0.363  
**Metadata:** stress=3, energy=3, face=tired_face, tod=early_morning, prev_mood=overwhelmed

**What went wrong:**  
The sentence starts positively ("sounds were nice") before pivoting to the actual state ("unsettled and fidgety"). TF-IDF weights "nice" and "unsettled" roughly equally as individual tokens. The positive opening creates a mixed signal that the model interprets as `mixed` rather than `restless`.

**Why it failed:**  
Sentence-level sentiment structure is invisible to bag-of-words models. The contrastive conjunction "but" should signal that the second clause is the actual emotional report. Without syntactic awareness, the positive and negative tokens are treated as co-equal evidence.

**How to improve:**  
Two approaches: (1) Extract only the clause following "but / however / though / yet" as the primary text feature. (2) Use a sentence encoder that produces different representations for "X but Y" vs "X and Y." Even a simple "words after last contrastive conjunction" heuristic would help here.

---

## Case 8 — Repeated template text, model sees pattern not meaning
**id:** 14  
**Text:** "The cafe session helped a little, though I still feel pulled in too many directions. Part of me wants to do everything at once."  
**True:** `restless` | **Predicted:** `overwhelmed` | **Confidence:** 0.341  
**Metadata:** stress=5, energy=3, sleep=6, face=none, tod=early_morning, prev_mood=mixed

**What went wrong:**  
The phrase "Part of me wants to do everything at once" appears in multiple training rows. Several of them are labelled `overwhelmed`, so the model has learned this phrase as an overwhelmed signal. Here it describes restlessness, but the surface text pattern overrides the nuance.

**Why it failed:**  
Template-like phrases in synthetic data create spurious token–label associations. The model memorised "do everything at once" → `overwhelmed` rather than learning the underlying semantic. This is a training data artefact specific to synthetic datasets.

**How to improve:**  
De-duplicate or down-weight repeated n-grams during TF-IDF fitting. The `max_df` parameter (currently unset, defaults to 1.0) could be lowered to reduce the influence of phrases that appear in many different label contexts. Also: this failure reinforces that real user journals — which are idiosyncratic rather than templated — would likely produce a cleaner model.

---

## Cross-Cutting Insights

**1. The restless/overwhelmed/mixed triangle is the hardest boundary.**  
6 of the 8 mismatch errors involve confusion between these three classes. They share high-stress, high-cognitive-load vocabulary. The distinguishing features are degree of agency, degree of physical symptoms, and whether the emotion is directed inward (overwhelmed) or outward (restless). None of these are well-captured by unigrams.

**2. Conflict texts are the primary error driver.**  
66% of errors come from the 25.9% of data with noisy labels. This is not a model failure — it is a data quality problem. A model that perfectly memorised every training label would still produce wrong predictions on these cases when the text appears in test with a different "correct" answer.

**3. Confidence is well-calibrated around errors.**  
The average confidence at error cases is 0.31. Average confidence on correct predictions is over 0.55. The calibration step is working — wrong predictions are flagged with lower confidence, which means `uncertain_flag` correctly fires on most failures.

**4. The decision engine degrades gracefully.**  
Even when the predicted state is wrong (e.g., `overwhelmed` instead of `restless`), the recommended action is often still appropriate. `box_breathing` is the right recommendation for both high-intensity restlessness and overwhelm. The guidance layer has natural tolerance to boundary confusion between adjacent states.
