# StanceEval2026 Modular Baseline

This repository now contains a modular Python implementation for Arabic target-specific stance classification with the fixed label mapping:

| Label | ID |
| --- | --- |
| Against | 0 |
| Favor | 1 |
| None | 2 |

The tokenizer input format is preserved from the original baseline:

```python
tokenizer(target, text, truncation=True, padding="max_length", max_length=128)
```

## Project Layout

```text
configs/config.yaml
src/data.py
src/preprocessing.py
src/models.py
src/losses.py
src/metrics.py
src/trainer.py
src/inference.py
src/utils.py
scripts/run_cls4_experiments.py
scripts/generate_submissions.py
scripts/smoke_test.py
kaggle_run.ipynb
```

The original `StanceEval2026_Baseline_Code.ipynb`, evaluation scripts, website assets, and MawqifV2 files are left in place.

## Data

By default, `configs/config.yaml` uses the dataset already included in this repository:

```text
MawqifV2/Track 1/train.csv
MawqifV2/Track 1/dev.csv
MawqifV2/Track 2/dev_track_2.csv
```

Required task columns are `id`, `target`, `text`, and `stance` for train/dev. The loader also accepts the current MawqifV2 `ID` column as an alias and writes submissions as `id,stance`.

If official `test_seen.csv` and `test_unseen.csv` files are present, the submission script uses them. If they are not present, it uses the repository fallback files so the notebook can run end-to-end without adding anything:

```text
submission_seen.csv   <- MawqifV2/Track 1/dev.csv
submission_unseen.csv <- MawqifV2/Track 2/dev_track_2.csv
```

For Kaggle, no manual path edit is needed when the whole project is uploaded and `kaggle_run.ipynb` is run from the project root. Optional path overrides are still supported:

```bash
export STANCEEVAL_DATA_DIR=/kaggle/working
export STANCEEVAL_OUTPUT_DIR=/kaggle/working/outputs
export STANCEEVAL_RESULTS_DIR=/kaggle/working/results
```

On Windows PowerShell:

```powershell
$env:STANCEEVAL_DATA_DIR="C:\path\to\data"
$env:STANCEEVAL_OUTPUT_DIR="outputs"
$env:STANCEEVAL_RESULTS_DIR="results"
```

## Compared Models

The configured workflow compares these base models with the same setup:

| Display name | Hugging Face ID |
| --- | --- |
| MARBERT | `UBC-NLP/MARBERT` |
| MARBERTv2 | `UBC-NLP/MARBERTv2` |
| ARBERT | `UBC-NLP/ARBERT` |
| AraBERT | `aubmindlab/bert-base-arabert` |
| AraBERTv2 | `aubmindlab/bert-base-arabertv2` |
| AraBERTv0.2 | `aubmindlab/bert-base-arabertv02` |
| AraBERTv0.2 Twitter | `aubmindlab/bert-base-arabertv02-twitter` |

## Experiments

The default workflow runs a fair base-model comparison:

- `base_final_cls_full_ft`: `AutoModelForSequenceClassification`, final-layer CLS, full fine-tuning.

Training uses the same baseline-style setup:

```yaml
max_length: 128
batch_size: 32
epochs: 20
learning_rate: 2e-5
early_stopping:
  enabled: true
  patience: 3
```

The best checkpoint is selected only by dev `Favg2` and copied to `outputs/best_model/`.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight check:

```bash
python scripts/smoke_test.py --config configs/config.yaml --base-model-id aubmindlab/bert-base-arabertv02-twitter --verbose
```

Run the full workflow:

```bash
python scripts/smoke_test.py --config configs/config.yaml --base-model-id aubmindlab/bert-base-arabertv02-twitter --verbose
python scripts/compare_base_models.py --config configs/config.yaml --verbose
python scripts/generate_submissions.py --config configs/config.yaml --verbose
```

## Kaggle Run

Use `kaggle_run.ipynb` and run all cells. It auto-detects Kaggle output paths and uses the included `MawqifV2` files. Equivalent commands:

```bash
pip install -q -r requirements.txt
export STANCEEVAL_DATA_DIR=/kaggle/working
export STANCEEVAL_OUTPUT_DIR=/kaggle/working/outputs
export STANCEEVAL_RESULTS_DIR=/kaggle/working/results
python scripts/smoke_test.py --config configs/config.yaml --base-model-id aubmindlab/bert-base-arabertv02-twitter --verbose
python scripts/compare_base_models.py --config configs/config.yaml --verbose
python scripts/generate_submissions.py --config configs/config.yaml --verbose
```

## Outputs

Base model comparison result:

```text
results/base_model_comparison.csv
```

Best final checkpoint:

```text
outputs/best_model/
```

Final submission files:

```text
results/submission_seen.csv
results/submission_unseen.csv
results/submission_files.zip
```

Each run directory stores training history, dev predictions, overall metrics, per-target metrics, a classification report, and a confusion matrix. Checkpoints for non-winning experiments are removed after final selection; their numeric results and predictions remain.
