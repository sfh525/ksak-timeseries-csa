# KSAK — HPC Workload Forecasting with Adaptive Input Normalization

This project compares **LSTM** forecasting setups for predicting HPC job submission intensity (`num_jobs` per 15-minute slice) from LLNL BlueGene/L Parallel Workloads Archive traces.

The goal is a controlled ablation of **input normalization**, with a shared experiment harness so differences in results can be attributed to the normalization method rather than data splits, architecture, or metrics.

## Task

- **Data:** SWF workload log, aggregated into fixed 15-minute slices with gap filling for idle periods.
- **Target:** next-slice `num_jobs` or `total_requested_proc`.
- **Features:** univariate (`num_jobs` or `total_requested_proc` only) or multivariate (job count plus allocated processors / run-time aggregates), selected via `FEATURE_MODE`.
- **Model backbone (all notebooks):** LSTM 64 → Dropout → LSTM 32 → Dropout → Dense 16 → Dense 1, trained with Adam.
- **Split:** chronological 70% / 15% / 15% (train / validation / test).
- **Lookback:** tuned once over `{30, 60, 90, 120}` slices using validation Iverson RMSE, then locked for final training.
- **Metrics:** Iverson-bracket I-MSE, I-RMSE, I-MAE, and I-R² (score only where actual `num_jobs ≠ 0`).

## Notebooks

| Notebook | Role | Input normalization | Target normalization |
|---|---|---|---|
| [`lstm-vanilla.ipynb`](lstm-vanilla.ipynb) | Baseline | Fixed per-window z-score on all input features | Approach A: raw per-window z-score of historical `num_jobs` |
| [`lstm-dain.ipynb`](lstm-dain.ipynb) | Adaptive (DAIN) | Deep Adaptive Input Normalization (`adaptive_scale`) inside the model | Same Approach A |
| [`lstm-edain.ipynb`](lstm-edain.ipynb) | Adaptive (EDAIN) | Train-only global `StandardScaler`, then Extended DAIN (outlier / shift / scale / power) | Same Approach A |

Shared harness details (all three):

- Boundary-safe sliding windows (validation/test may use prior-split history).
- EarlyStopping + ReduceLROnPlateau.
- Same SWF preprocess, feature switch, window candidates, batch size, and epoch budget.

Intentional method differences:

- **Vanilla** uses a non-learned per-window input z-score.
- **DAIN** learns adaptive mean/scale transforms on each input window (separate DAIN learning rates).
- **EDAIN** follows the global-aware recipe: fixed global z-score on inputs, then learned EDAIN sublayers (separate LR scales for outlier/shift/scale/power).

Target space is aligned across notebooks (Approach A) so comparisons isolate **input** normalization.

## How to run

1. Open a notebook on Kaggle (or locally) with the SWF dataset available at the path set in the notebook.
2. Set `FEATURE_MODE` to `'univariate'` or `'multivariate'`.
3. Run all cells top to bottom (window tuning trains several short models; final training follows).

Expected outputs: selected lookback window, training/validation curves, test prediction overlay, and an Iverson metrics table.
