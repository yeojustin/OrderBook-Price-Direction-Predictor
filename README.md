## Order Book Price Movement Predictor

This project predicts short-term BTC price direction from Binance Level-2 (top 5 levels) order book data.

It is organized as a simple 3-stage workflow:

1. **Ingest live order book data** from Binance WebSocket
2. **Transform raw snapshots** into model-ready features + labels
3. **Train and run inference** with baseline ML models

---

## Repository Structure

- `ingest_data/`
  - `collector.py` - BTC data collector script
  - `collector2.py` - ETH data collector script
  - `data/` - raw collected CSV files
- `transform_data/`
  - `transform.ipynb` - feature engineering and target creation notebook
  - `l2_data_btcusdt.csv` - raw BTC dataset used by transform notebook
  - `l2_data_btcusdt_transformed.csv` - transformed dataset for training
- `model_training/`
  - `train.ipynb` - model training and evaluation notebook
  - `predict.py` - command-line inference script
  - `artifacts/` - saved model artifacts (`.joblib`)
- `requirements.txt` - pip dependencies
- `pyproject.toml` / `uv.lock` - project metadata and lockfile

---

## What Data Looks Like

Each raw row is one snapshot with:

- `timestamp`
- `mid_price`
- bid prices/quantities: `b_p_0..4`, `b_q_0..4`
- ask prices/quantities: `a_p_0..4`, `a_q_0..4`

The transformed dataset adds engineered features such as:

- `obi` (order book imbalance)
- `spread`
- `depth_skew`
- `momentum_10s`
- `future_mid_price`
- `target` (1 = up, 0 = down/flat over prediction window)

---

## Setup

### Option A: Use existing virtual environment

If you already have `.venv` in this repo:

```bash
source .venv/bin/activate
```

### Option B: Create environment with pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) Collect Data

From repo root:

```bash
python ingest_data/collector.py
```

or:

```bash
python ingest_data/collector2.py
```

Stop with `Ctrl+C`.

Notes:

- `collector.py` and `collector2.py` are currently separate single-symbol scripts.
- Output is CSV under `ingest_data/data/` (or script-relative `data/` depending on where you run from).

---

## 2) Transform Data

Open and run:

- `transform_data/transform.ipynb`

This notebook:

- loads raw BTC CSV
- converts columns to numeric
- creates time index and sorts by time
- engineers features (`obi`, `spread`, `depth_skew`, `momentum_10s`)
- creates the classification target using a forward time window
- writes/uses transformed output CSV for modeling

Expected training input file:

- `transform_data/l2_data_btcusdt_transformed.csv`

---

## 3) Train Models

Open and run:

- `model_training/train.ipynb`

What it does:

- builds feature matrix + target
- applies chronological train/test split
- trains baseline models:
  - Logistic Regression
  - Random Forest
- reports metrics (`accuracy`, `F1`, `ROC-AUC`)
- runs walk-forward validation (`TimeSeriesSplit`)
- tunes decision threshold by F1
- saves artifact to:
  - `model_training/artifacts/btc_direction_model.joblib`

---

## 4) Run Inference

Use the saved artifact with:

```bash
python model_training/predict.py
```

Default behavior:

- loads artifact from `model_training/artifacts/btc_direction_model.joblib`
- loads input from `transform_data/l2_data_btcusdt_transformed.csv`
- scores the last valid row
- prints JSON:
  - `row_index`
  - `decision_threshold`
  - `proba_up`
  - `pred_up`

Useful options:

```bash
python model_training/predict.py --row 100
python model_training/predict.py --input-csv path/to/file.csv
python model_training/predict.py --artifact path/to/model.joblib
```

### `predict.py` Quick Reference

| Item | Default | Notes |
|---|---|---|
| `--artifact` | `model_training/artifacts/btc_direction_model.joblib` | Trained model + threshold metadata |
| `--input-csv` | `transform_data/l2_data_btcusdt_transformed.csv` | Must contain required feature columns |
| `--row` | `-1` | `-1` = latest valid row, or pass a specific index |
| Output: `row_index` | - | Scored row index |
| Output: `decision_threshold` | from artifact | Cutoff used for class decision |
| Output: `proba_up` | - | Probability of class `1` (up) |
| Output: `pred_up` | - | Final class (`1` up, `0` down/flat) |

---

## End-to-End Flow

1. Run collector to gather fresh market data
2. Copy/point the raw CSV into `transform_data/`
3. Run `transform_data/transform.ipynb`
4. Run `model_training/train.ipynb`
5. Run `model_training/predict.py`

---

## Troubleshooting

- **`Artifact not found`**
  - Run `model_training/train.ipynb` first to create the artifact.
- **Missing feature columns during inference**
  - Ensure input CSV is the transformed dataset, not raw collector output.
- **Notebook import/module errors**
  - Activate `.venv` and reinstall requirements.
- **Unstable metrics**
  - This is expected with short windows/high noise. Use more data and walk-forward checks.

---

## Next Steps (... I am still implementing)

- Live prediction from websocket feed to model
- Merge collectors into one configurable multi-symbol script
- Add automated transform + train pipeline script
- Add backtesting with threshold and transaction cost assumptions
- Add model versioning and experiment tracking
