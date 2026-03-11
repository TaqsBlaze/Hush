![icon](https://raw.githubusercontent.com/TaqsBlaze/Hush/refs/heads/main/image/image.png)

# Hush

Hush is a research-grade text classifier that flags toxic language in long-form messages by combining character-level TF-IDF extraction with a robust linear classifier. The project is tuned for clarity of metrics, reproducible training, and simple deployment, making it easy for moderators, educators, or open-source contributors to iterate on custom rules or datasets.

## Highlights
- **Character-aware embedding**: `TfidfVectorizer` runs on `char_wb` n-grams (3–5 characters) so the model catches insults that span creative spellings or leetspeak.
- **Balanced linear model**: `SGDClassifier` with `modified_huber` loss and class weights keeps training fast, stable, and sensitive to the minority toxic class.
- **Versioned artifacts**: Each training run writes timestamped models, vectorizers, and metadata plus `latest` copies for quick inference.

## Datasets & Generated Content
- `classification_data.csv` is the primary labeled corpus (toxic=1, non-toxic=0) that `trainer.py` consumes. The dataset mixes real and synthetic sentences and already includes an 80/20 split inside the training workflow.
- `classification_data-shona.csv` mirrors that labeling format but covers Shona-language statements to help evaluate multilingual generalization.
- `generated_5000_dataset.csv` is produced by `data_generator.py`, which stitches together templates for supportive and toxic phrasing. Re-run the generator to refresh the synthetic pool when you need more training examples.
- `classification_data-old.csv` and the metadata JSON files (e.g., `metadata_v20260311_011709.json`) document prior runs or auxiliary exports.

## Getting Started
1. Install the required packages:
   ```
   pip install pandas scikit-learn joblib
   ```
2. Adjust `classification_data.csv` (or swap in `generated_5000_dataset.csv`) as needed.
3. Run the training script to produce fresh artifacts.

## Training (`trainer.py`)
```bash
python trainer.py
```
- Loads the chosen CSV, drops NaNs, and stratifies into an 80/20 train/test split using `train_test_split(random_state=42)`.
- Configures `TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=50000)` and fits/transforms the text.
- Fits `SGDClassifier(loss="modified_huber", penalty="l2", alpha=0.0001, class_weight="balanced", random_state=42)` on the vectorized training set.
- Computes accuracy and a classification report on the test split, then saves:
  - `toxic_model_v<timestamp>.hush`
  - `vectorizer_v<timestamp>.hush`
  - `metadata_v<timestamp>.json` (contains accuracy, precision/recall for the toxic label, and training params)
  - `toxic_model_latest.hush` / `vectorizer_latest.hush` (overwrites with the newest run)

## Evaluation (`test_model.py`)
```bash
python test_model.py
```
- Loads the versioned artifacts referenced at the top of the script (update the filenames if you retrain with new timestamps).
- Runs through curated test cases that cover non-toxic, obviously toxic, subtle toxicity, and edge cases.
- Prints a simple table with pass/fail status plus overall percentage correct.

## Inference (`model.py`)
```bash
python model.py "Your message here"
# or run without arguments to use the interactive prompt
```
- Loads the artifacts hard-coded near the top; swap those filenames after retraining.
- Transforms the user text and prints whether Hush considers it toxic.

## Supporting Scripts
- `data_generator.py` regenerates a balanced (2,500/2,500) dataset of synthetic sentences with both polite and aggressive language. Run it to refresh `generated_5000_dataset.csv` or to seed new labels.
- Keep `README.md`, `FIX.md`, and `metadata_*.json` up to date whenever you change the training pipeline so contributors can track regressions.

## Artifact Reference
| File | Purpose |
| --- | --- |
| `toxic_model_v<TIMESTAMP>.hush` | Versioned classifier for reproducibility. |
| `vectorizer_v<TIMESTAMP>.hush` | Matching vectorizer used during training. |
| `metadata_v<TIMESTAMP>.json` | Stores metrics, parameters, and dataset provenance. |
| `toxic_model_latest.hush`, `vectorizer_latest.hush` | Handy shortcuts for inference. |
| `generated_5000_dataset.csv` | Output of `data_generator.py`, useful as supplemental training data. |

## License
Hush is MIT-licensed. See `LICENSE` for the full text.
