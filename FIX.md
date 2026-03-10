## Required Fixes

1. **Broaden training coverage**
   - Augment `classification_data.csv` with longer, context-rich toxic and non-toxic sentences so the model learns to weigh toxic cues even when embedded in otherwise benign prose.
   - Ensure both labels appear in long-form examples (e.g., multi-clause scenarios, indirect insults) instead of just short standalone slurs.

2. **Relax vectorization limits**
   - Remove or raise `TfidfVectorizer(max_features=1500)` so less frequent but critical terms (like “ass” in a long sentence) stay in the feature set. Consider adding `analyzer='char_wb'` with a wider `ngram_range` to capture substring-level toxicity.
   - Keep `stop_words='english'` but evaluate whether certain words should be preserved to retain context (e.g., negations or intensifiers).

3. **Add supplementary features or models**
   - Introduce auxiliary features (keyword counts, sentiment scores, sentence length) so the classifier can better identify toxic signals spread across long texts.
   - Optionally switch from `SGDClassifier` to a transformer-based approach (e.g., fine-tune a Hugging Face model) or add a pipeline that scores sliding windows and aggregates the highest toxicity score.

4. **Improve evaluation coverage**
   - Create test cases (or augment `test_model.py`) with long inputs like “I met a guy…” to verify the model now predicts toxicity for those scenarios.
   - Log misclassified long examples during training to identify remaining blind spots and inform further data additions.

5. **Persist updated assets**
   - After retraining with the new vectorizer/features, regenerate `toxic_model.hush` and `vectorizer.hush`.
   - Keep version history or changelog entries if necessary for reproducibility.
