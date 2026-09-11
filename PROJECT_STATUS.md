# SIH26153 Network Attack Forecasting — Project Status

## Phase 1: Data Pipeline ✅ COMPLETE
- **Pipeline**: `src/pipeline/` — load_cicids2017, apply_stage_labels, select_features, normalise, reconstruct_campaigns, create_windows
- **Data**: 100K rows subsampled from CIC-IDS2017 (2.8M total)
- **Sequences**: 97,661 sequences, shape (97661, 20, 46)
- **Campaigns**: 100 multi-stage campaigns with shuffled flows
- **Memory**: RobustScaler normalisation, float32 throughout
- **Checkpoint**: `data/sequences/X.npy`, `y.npy`, `metadata.json` ✅

## Phase 2: Markov Baseline ✅ COMPLETE
- **File**: `src/baseline/markov.py`
- **Model**: First-order Markov chain (7×7 transition matrix)
- **Accuracy**: 80.15% (predicts Benign most of the time)
- **Artifact**: `models/markov_baseline.npz`, `models/markov_metadata.json`

## Phase 3: LSTM Model ✅ COMPLETE
- **File**: `src/models/lstm.py`
- **Architecture**: LSTM(46→128, 2 layers, dropout=0.3, FC→7)
- **Parameters**: 223,111
- **Accuracy**: 80.15% (converges to Benign prediction)
- **Artifact**: `models/lstm_best.pth`, `models/lstm_metadata.json`
- **Note**: Model converges to predicting Benign (80% majority class). This is expected given the dataset skew.

## Remaining Phases
- Phase 4: Transformer model
- Phase 5: Explainability (SHAP/LIME)
- Phase 6: FastAPI backend
- Phase 7: React dashboard
- Phase 8: Evaluation suite
- Phase 9: Demo hardening
- Phase 10: Tests

## Key Constants
- STAGE_TO_IDX: {Benign:0, Recon:1, CredAccess:2, Exploit:3, LateralMove:4, C2:5, Impact:6}
- WINDOW_SIZE: 20
- FEATURE_COLS: 46 features
- Model directory: `models/`
