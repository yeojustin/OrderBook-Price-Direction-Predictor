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

Single symbol scripts (legacy):

```bash
python ingest_data/collector.py
python ingest_data/collector2.py
```

Multi-symbol collector (recommended):

```bash
python ingest_data/collector_multi.py --symbols btcusdt,ethusdt
```

Useful options:

```bash
python ingest_data/collector_multi.py --symbols btcusdt,ethusdt,solusdt --depth-levels 5 --interval-ms 1000
```

Output files are written under `ingest_data/data/` as `l2_data_<SYMBOL>.csv`.

---

## 2) Transform Data

Notebook version:

- `transform_data/transform.ipynb`

Script version (recommended for repeatable runs):

```bash
python transform_data/transform_dataset.py \
  --input-csv transform_data/l2_data_btcusdt.csv \
  --output-csv transform_data/l2_data_btcusdt_transformed.csv
```

This step creates:

- `obi`
- `spread`
- `depth_skew`
- `momentum_10s`
- `future_mid_price`
- `target`

---

## 3) Train Models

Notebook version:

- `model_training/train.ipynb`

Script version:

```bash
python model_training/train_model.py \
  --input-csv transform_data/l2_data_btcusdt_transformed.csv \
  --artifact-dir model_training/artifacts
```

Training script outputs:

- versioned artifact: `model_training/artifacts/btc_direction_model_<timestamp>.joblib`
- latest artifact alias: `model_training/artifacts/btc_direction_model.joblib`
- metrics JSON: `model_training/artifacts/metrics_<timestamp>.json`

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

## 5) Live Prediction (WebSocket + Model)

Run live inference directly from Binance depth stream:

```bash
python live_prediction/live_predict.py --symbol btcusdt
```

Example with options:

```bash
python live_prediction/live_predict.py \
  --symbol ethusdt \
  --artifact model_training/artifacts/btc_direction_model.joblib \
  --depth-levels 5 \
  --interval-ms 1000 \
  --momentum-periods 10
```

The script prints one JSON line per prediction with `proba_up` and `pred_up`.

---

## 6) Backtesting with Trading Costs

```bash
python model_training/backtest.py \
  --artifact model_training/artifacts/btc_direction_model.joblib \
  --input-csv transform_data/l2_data_btcusdt_transformed.csv \
  --fee-bps 1.0
```

Optional threshold override:

```bash
python model_training/backtest.py \
  --input-csv transform_data/l2_data_btcusdt_transformed.csv \
  --threshold 0.55
```

---

## 7) One-Command Transform + Train Pipeline

```bash
python run_pipeline.py \
  --raw-csv transform_data/l2_data_btcusdt.csv \
  --transformed-csv transform_data/l2_data_btcusdt_transformed.csv \
  --artifact-dir model_training/artifacts
```

This runs:

1. `transform_data/transform_dataset.py`
2. `model_training/train_model.py`

---

## End-to-End Flow

1. Run `ingest_data/collector_multi.py` to gather market data
2. Transform with `transform_data/transform_dataset.py`
3. Train with `model_training/train_model.py` (or `run_pipeline.py`)
4. Score batches with `model_training/predict.py`
5. Run real-time scoring with `live_prediction/live_predict.py`
6. Evaluate strategy with `model_training/backtest.py`

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

## Remaining Next Steps

- Add richer backtesting logic (position sizing, hold horizon, latency assumptions)
- Add automated periodic retraining
- Add model registry/experiment tracking
