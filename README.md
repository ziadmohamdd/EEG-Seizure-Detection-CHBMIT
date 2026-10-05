# EEG Seizure Detection — CHB-MIT Multi-Model Pipeline

Deep-learning pipeline for automatic seizure detection on the [CHB-MIT Scalp EEG
Database](https://physionet.org/content/chbmit/1.0.0/) (24 pediatric patients).
Six architectures are trained and compared across two feature sets (EEG-only vs.
EEG + ECG proxy) and three optimizers each.

## Pipeline Overview

```
01_Phase1_Preprocessing_FIXED.ipynb   → raw .edf → windowed band-power features
02_Utils_ModelUtils_FIXED.ipynb       → shared helpers (reference; not imported by model notebooks)
03_Model1_CNN2D_FIXED.ipynb           → 2D CNN
04_Model2_Conformer_FIXED.ipynb       → Conformer (conv-augmented transformer)
05_Model3_LSTM_FIXED.ipynb            → Bidirectional LSTM
06_Model4_CNN_LSTM_FIXED.ipynb        → Hybrid CNN + LSTM
07_Model5_LSTM_Conformer_FIXED.ipynb  → Hybrid LSTM + Conformer
08_Model6_ComplexHybrid_FIXED.ipynb   → 3-branch CNN + LSTM + Conformer fusion
09_Phase3_ResultsCompilation.ipynb    → aggregates results.csv into a leaderboard
```

Each model notebook trains on **Dataset A** (EEG band-power features, 18 channels
× 5 frequency bands) and **Dataset B** (Dataset A + an ECG proxy channel), each
with three optimizers (Adam, Nadam, AdamW) — 6 trials per notebook, 36 total.
Every trial extracts a deep feature embedding and evaluates it downstream with
an RBF-kernel SVM.

## Setup

1. Download CHB-MIT from PhysioNet and place it on Google Drive at
   `MyDrive/Physionet.org/files/chbmit/1.0.0/`.
2. Run `01_Phase1_Preprocessing_FIXED.ipynb` in Colab (mounts Drive, writes
   `dataset_A.csv` / `dataset_B.csv` to `MyDrive/epilepsy_outputs/`).
   Expect ~1.25-1.5 hours for all 24 patients.
3. Run `03`-`08` in any order (each is independent, reads the same two CSVs).
   Each notebook auto-resumes from its last completed epoch/trial if the Colab
   runtime disconnects — no manual intervention needed on rerun.
4. Run `09_Phase3_ResultsCompilation.ipynb` once all 6 are done to produce
   `leaderboard.xlsx`.

## Key Fixes Applied (vs. original pipeline)

A number of real bugs were found and fixed during development — worth noting
for reproducibility and for anyone adapting this pipeline:

- **Channel-schema drift**: per-patient feature columns varied when a channel
  was missing from a recording, corrupting the combined CSV. Fixed by
  reindexing every patient onto a fixed 18-channel × 5-band schema.
- **`T8-P8` duplicate channel**: CHB-MIT contains two channels both literally
  named `T8-P8`; MNE auto-suffixes them (`T8-P8-0`/`T8-P8-1`) on load, which
  broke matching against the standard channel list — silently zero-filling
  this channel for every window, every patient. Fixed via alias remapping.
- **ECG channel misidentification**: the intended ECG proxy channel name
  matched almost no recordings; a full-corpus channel tally found the real
  cardiac channel is labeled plain `ECG` in most cases.
- **Training collapse (AUC stuck at 0.5)**: caused by unnormalized input
  features and lack of output-bias initialization under extreme class
  imbalance (~0.19% positive rate). Fixed with train-only feature
  normalization and a prior-informed output bias.
- **Conformer-family instability**: models containing Conformer blocks
  (LayerNorm-heavy) were prone to the same collapse even with the above
  fixes, at the default learning rate / bias magnitude. Mitigated with a
  capped output bias and a lower learning rate for those architectures.
- **SVM evaluation bottleneck**: RBF-kernel SVC does not scale to ~700K
  training rows; subsampled to 20K rows for the downstream SVM evaluation
  step only (deep model training is unaffected).
- **Epoch-level resume**: training checkpoints a `_latest` model every epoch
  (not just on AUC improvement), so a Colab disconnect loses at most one
  epoch of progress rather than an entire trial.

## Results

See `results.csv` / `leaderboard.xlsx` for the full 36-trial table. Summary —
best AUC per architecture:

| Model | Best AUC | Dataset / Optimizer |
|---|---|---|
| CNN2D | 0.876 | A / AdamW |
| Conformer | 0.845 | A / Adam |
| LSTM | 0.969 | B / Adam |
| CNN + LSTM | 0.989 | B / AdamW |
| LSTM + Conformer | 0.985 | B / Nadam |
| Complex Hybrid | *see results.csv* | — |

A consistent pattern across all evaluated models: **Dataset B (EEG + ECG
proxy) outperformed Dataset A (EEG-only)** despite a much smaller sample
(164K vs. ~1.06M windows), and **recurrent/hybrid architectures outperformed
pure CNN or attention-only models**.

## Data

Raw CHB-MIT `.edf` files and the generated `dataset_A.csv` / `dataset_B.csv`
are not included in this repository (size / licensing). Regenerate them with
`01_Phase1_Preprocessing_FIXED.ipynb` after downloading the dataset from
PhysioNet (link above).

## Notes

- `09_Phase3_TrainingMatrix.ipynb` (an earlier, unfixed single-notebook
  re-implementation of all 6 models) is intentionally excluded — superseded
  by the individually-fixed notebooks `03`-`08`.
- Trained model checkpoints (`.keras` files) are not included; see Setup to
  reproduce, or request weights separately.
