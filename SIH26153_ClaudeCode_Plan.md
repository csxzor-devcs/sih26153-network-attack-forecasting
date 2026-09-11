# SIH26153 — Claude Code Master Build Plan
## AI-based Network Attack Forecasting from Network Traffic Data (NTRO)

> **How to use this file**: Feed each Phase Prompt to Claude Code in order.
> Complete the ✅ checkpoint before moving to the next phase.
> Never skip a phase — each one is a dependency for the next.

---

## Project Structure (Claude Code will create this)

```
sih26153/
├── data/
│   ├── raw/                  # Downloaded CIC-IDS CSVs go here (manual)
│   ├── processed/            # Cleaned + stage-labelled flows
│   └── sequences/            # Sliding-window sequences for training
├── src/
│   ├── config.py             # Stage map, MITRE map, all constants
│   ├── pipeline/
│   │   ├── loader.py         # CSV loading + merging
│   │   ├── labeller.py       # Dataset label → attack stage mapping
│   │   ├── sequencer.py      # Campaign reconstruction + windowing
│   │   └── features.py       # Feature selection + normalisation
│   ├── models/
│   │   ├── markov.py         # Baseline Markov chain
│   │   ├── lstm_model.py     # LSTM sequence forecaster
│   │   ├── transformer_model.py  # Transformer encoder forecaster
│   │   └── train.py          # Training loop + evaluation
│   ├── explainability/
│   │   ├── attention.py      # Attention rollout extractor
│   │   └── shap_explainer.py # SHAP values on flow features
│   └── api/
│       ├── main.py           # FastAPI app
│       ├── schemas.py        # Pydantic request/response models
│       └── replay.py         # Attack campaign replay engine
├── dashboard/                # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── LiveFeed.jsx      # Streaming flow feed
│   │   │   ├── ForecastPanel.jsx # Next-stage probability bars
│   │   │   ├── MitreMap.jsx      # ATT&CK tactic display
│   │   │   └── ExplainPanel.jsx  # Causative flow highlight
│   │   └── api/client.js
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sequence_analysis.ipynb
│   └── 03_model_comparison.ipynb
├── tests/
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_api.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

## BEFORE YOU START — Manual step (Claude Code cannot do this)

Download the datasets yourself:

1. **CIC-IDS2017** → https://www.unb.ca/cic/datasets/ids-2017.html
   - Download the "MachineLearningCSV.zip" (pre-extracted flows, ~500MB)
   - Place all CSVs inside `data/raw/cicids2017/`

2. **UNSW-NB15** → https://research.unsw.edu.au/projects/unsw-nb15-dataset
   - Download the 4 CSV files
   - Place inside `data/raw/unswnb15/`

Once downloaded, run Phase 1.

---

## Phase 0 — Scaffold & Environment

**Paste this into Claude Code:**

```
Create the full project scaffold for SIH26153 — AI-based Network Attack Forecasting.

Project root: sih26153/

Tasks:
1. Create ALL directories as listed in the project structure above
2. Create requirements.txt with:
   - torch>=2.2.0
   - pandas>=2.0.0
   - numpy>=1.26.0
   - scikit-learn>=1.4.0
   - matplotlib>=3.8.0
   - seaborn>=0.13.0
   - shap>=0.44.0
   - fastapi>=0.110.0
   - uvicorn>=0.28.0
   - pydantic>=2.0.0
   - jupyter>=1.0.0
   - tqdm>=4.66.0
   - scipy>=1.12.0
3. Create src/config.py with:
   - STAGE_MAP: dict mapping CIC-IDS2017 label strings to stage names
     (BENIGN→"Benign", PortScan→"Recon", FTP-Patator→"CredAccess",
      SSH-Patator→"CredAccess", DoS Hulk→"Impact", DoS GoldenEye→"Impact",
      DoS slowloris→"Impact", DoS Slowhttptest→"Impact",
      Heartbleed→"Exploit", Web Attack – Brute Force→"CredAccess",
      Web Attack – XSS→"Exploit", Web Attack – Sql Injection→"Exploit",
      Infiltration→"LateralMove", Bot→"C2", DDoS→"Impact")
   - MITRE_MAP: dict mapping stage names to MITRE ATT&CK tactic IDs and names
     (Recon→TA0043 Reconnaissance, CredAccess→TA0006 Credential Access,
      Exploit→TA0001 Initial Access, LateralMove→TA0008 Lateral Movement,
      C2→TA0011 Command and Control, Impact→TA0040 Impact)
   - STAGE_ORDER: list defining natural progression order
   - WINDOW_SIZE: 20 (sliding window length)
   - FEATURE_COLS: list of the 20 most informative CIC-IDS features (flow duration,
     packet lengths, IAT stats, flag counts, bytes per second, etc.)
   - MODEL_DIR: Path to saved models
   - SEQUENCE_DIR: Path to processed sequences
4. Create a README.md with project description, setup instructions, and the
   forecasting-vs-classification distinction explained clearly
5. Create docker-compose.yml with services for: api (FastAPI) and dashboard (React/Nginx)
6. Install dependencies by running: pip install -r requirements.txt

Verify by printing the directory tree after creation.
```

**✅ Checkpoint:** `sih26153/` directory exists, `pip install` completes without errors, `src/config.py` is importable.

---

## Phase 1 — Data Pipeline

**Paste this into Claude Code:**

```
Working in sih26153/src/pipeline/, build the complete data loading and preprocessing pipeline.

The raw CIC-IDS2017 CSVs are in data/raw/cicids2017/. Column names have leading spaces — strip them.

Build these files:

1. loader.py
   - load_cicids2017(data_dir) → pd.DataFrame
     * Load all CSVs in the directory, concatenate
     * Strip whitespace from column names and string values
     * Drop rows where Label is NaN
     * Convert Timestamp to datetime, sort ascending
     * Print shape and Label value_counts() after loading
   - load_unswnb15(data_dir) → pd.DataFrame (similar, different column names)
   - merge_datasets(df1, df2) → pd.DataFrame with a 'dataset_source' column

2. labeller.py
   - apply_stage_labels(df) → df with new column 'stage' using STAGE_MAP from config.py
   - apply_mitre_labels(df) → df with columns 'mitre_id' and 'mitre_tactic'
   - get_stage_distribution(df) → dict with stage counts and percentages
   - Print a clear table showing: original label → stage → MITRE tactic for each unique label

3. features.py
   - select_features(df) → df with only FEATURE_COLS + metadata cols (timestamp, src IP, stage)
   - handle_infinities(df) → replace inf/-inf with column max/0
   - handle_nulls(df) → fill NaN with column median
   - normalise(df, scaler=None) → (normalised_df, fitted_scaler)
     * Use RobustScaler (better for network traffic distributions)
     * If scaler is None, fit a new one. Otherwise use the provided one.
   - Save the fitted scaler to data/processed/scaler.pkl

4. sequencer.py
   - reconstruct_campaigns(df) → dict mapping (src_ip) → list of (timestamp, stage, features)
     * Group by Source IP
     * Sort each group by timestamp
     * Filter: keep only IPs with 2+ distinct stages (these are multi-stage attacks)
   - create_windows(campaigns, window_size=WINDOW_SIZE) → list of (X, y) tuples
     * X: numpy array shape (window_size, n_features) — the last WINDOW_SIZE flows
     * y: integer stage label of the NEXT flow after the window
     * Slide the window by 1 flow at a time
     * Skip windows where all stages in X are the same as y (no progression to predict)
   - save_sequences(windows, output_dir) → saves X.npy, y.npy, metadata.json
   - print_campaign_stats(campaigns) → show: total campaigns, avg length, stage transition matrix

After building all files, create a script pipeline/run_pipeline.py that:
- Loads CIC-IDS2017
- Applies stage labels
- Selects + normalises features
- Reconstructs campaigns
- Creates windows
- Saves to data/sequences/
- Prints a summary: "Created N sequences from M campaigns. Stage distribution: ..."

Run it and show the output.
```

**✅ Checkpoint:** `data/sequences/X.npy`, `y.npy`, `metadata.json` exist. At least 500 multi-stage sequences created. Stage transition matrix printed showing Recon → Exploit type progressions.

---

## Phase 2 — Baseline Model (Markov Chain)

**Paste this into Claude Code:**

```
Build the Markov chain baseline in sih26153/src/models/markov.py.

This is the "dumb" baseline. The LSTM must beat it. Build it properly so it gives a fair fight.

Class: MarkovForecaster
  - __init__(self, order=1)  # order=2 means bigram transitions
  - fit(self, sequences)
    * sequences: list of lists of stage labels (strings)
    * Build transition probability matrix: P(next_stage | current_stage)
    * For order=2: P(next_stage | stage_t-1, stage_t)
    * Store as nested dict
  - predict_proba(self, stage_sequence) → dict mapping stage → probability
    * Given a sequence of stages, return prob distribution over next stage
    * Fall back to uniform distribution if the transition was never seen in training
  - predict(self, stage_sequence) → str (most likely next stage)
  - evaluate(self, X_sequences, y_true) → dict with metrics:
    * accuracy: fraction where predict() == y_true
    * top2_accuracy: fraction where y_true is in top-2 predictions
    * mean_forecast_confidence: average probability assigned to correct stage
    * per_stage_precision: dict of precision for each stage
  - plot_transition_matrix(self, save_path) → heatmap of P(next|current)

Also add to markov.py:
  - baseline_random(y_true) → accuracy of random uniform predictor
  - baseline_majority(y_true) → accuracy of always predicting most-common stage

Create notebooks/01_data_exploration.ipynb with cells that:
1. Load the processed sequences
2. Show stage distribution bar chart
3. Plot a sample campaign timeline (x=flow index, y=stage, colored by stage)
4. Show the Markov transition matrix heatmap
5. Print Markov baseline accuracy vs random vs majority

Run the notebook and show the final accuracy numbers for all three baselines.
```

**✅ Checkpoint:** Markov accuracy printed. Should be meaningfully above random (random ≈ 1/6 ≈ 17%). If Markov is below 35%, the sequence construction has a bug — go back to Phase 1.

---

## Phase 3 — LSTM Forecaster

**Paste this into Claude Code:**

```
Build the LSTM sequence forecaster in sih26153/src/models/lstm_model.py and train.py.

lstm_model.py:
  Class: AttentionLSTM(nn.Module)
    - __init__(self, input_size, hidden_size=128, num_layers=2, num_stages=6, dropout=0.3)
    - Forward pass:
      * LSTM over input sequence → hidden states shape (T, batch, hidden)
      * Additive attention: score each timestep, softmax → attention weights shape (T, batch, 1)
      * Context vector: weighted sum of hidden states
      * FC layer: context → logits over num_stages
    - get_attention_weights(self, x) → attention weights tensor (T, batch)
      * This is critical for explainability — implement it cleanly
    - forward returns: (logits, attention_weights)

  Class: ForecastDataset(Dataset)
    - __init__(self, X_path, y_path)
      * Load X.npy (shape: N, T, features) and y.npy (shape: N,)
      * Convert to float32 tensors
    - __len__, __getitem__

train.py:
  - train_epoch(model, loader, optimizer, criterion) → avg_loss, avg_acc
  - eval_epoch(model, loader, criterion) → avg_loss, avg_acc, per_stage_metrics
  - train_model(config_dict) → trained model + training history
    * config: hidden_size, num_layers, dropout, lr, batch_size, epochs, device
    * Use Adam optimizer, CosineAnnealingLR scheduler
    * Use CrossEntropyLoss with class weights (stages are imbalanced)
    * Save best model by val accuracy to MODEL_DIR/best_lstm.pt
    * Save training history to MODEL_DIR/lstm_history.json
  - evaluate_model(model, test_loader) → full metrics dict:
    * accuracy, top2_accuracy
    * per_stage_precision, recall, f1
    * mean_forecast_lead_time: avg number of flows BEFORE the real stage change
      that the model first predicts the correct next stage above 0.5 confidence
      THIS is the key SIH metric — we want this to be 3+ flows early
    * confusion_matrix
  - plot_training_curves(history, save_path)
  - plot_confusion_matrix(cm, save_path)

Training config to use:
  hidden_size=128, num_layers=2, dropout=0.3, lr=1e-3,
  batch_size=64, epochs=30, device='cuda' if available else 'cpu'

After building, train the model and print:
  - Final val accuracy
  - Final test accuracy
  - Mean forecast lead time (in flows)
  - Per-stage F1 scores
  - "Beats Markov by X%: YES/NO"

Save all plots to notebooks/figures/.
```

**✅ Checkpoint:** Test accuracy > Markov baseline. Mean forecast lead time ≥ 2 flows. Model saved to `MODEL_DIR/best_lstm.pt`. If accuracy < 40%, check class weights and sequence quality.

---

## Phase 4 — Transformer Forecaster

**Paste this into Claude Code:**

```
Build the Transformer forecaster in sih26153/src/models/transformer_model.py.

This is the upgrade over LSTM. Use the same training infrastructure from train.py.

Class: AttackTransformer(nn.Module)
  - __init__(self, input_size, d_model=128, nhead=4, num_encoder_layers=3,
              num_stages=6, dropout=0.1, max_seq_len=50)
  - Architecture:
    * Linear projection: input_size → d_model
    * Positional encoding (sinusoidal, learned positions)
    * Transformer encoder (d_model, nhead, num_encoder_layers)
    * CLS token approach: prepend a learnable [CLS] token, use its output for classification
    * FC: d_model → num_stages
  - get_attention_maps(self, x) → attention weights from all heads, all layers
    * Return shape: (num_layers, num_heads, T+1, T+1)
    * Use register_forward_hook to capture attention weights during forward pass
  - forward returns: (logits, attention_maps)

Add to train.py:
  - train_transformer(config_dict) → trained model
    * Same structure as train_model but saves to MODEL_DIR/best_transformer.pt
  - compare_models(lstm_metrics, transformer_metrics) → print comparison table

Also create notebooks/03_model_comparison.ipynb:
  - Load both models
  - Run both on the test set
  - Side-by-side table: accuracy, lead time, per-stage F1, inference speed
  - Plot: for a sample attack campaign, show both models' probability outputs
    over time as the attack progresses (x=flow index, y=prob of correct next stage)
  - Conclusion cell: which model to use in the demo and why

Run the comparison and show the table.
```

**✅ Checkpoint:** Transformer trained and compared. Pick the better model (usually Transformer by ~3-5%). Note which one you'll use for the demo.

---

## Phase 5 — Explainability

**Paste this into Claude Code:**

```
Build the explainability layer in sih26153/src/explainability/.

attention.py:
  - extract_attention_rollout(model, x) → attention weights shape (T,)
    * For Transformer: implement attention rollout (Abnar & Zuidema 2020)
      - Multiply attention matrices across layers, account for residual connections
    * For LSTM: use the additive attention weights directly
    * Returns: importance score per timestep (flow) in the input window
  - get_top_k_flows(attention_weights, flow_metadata, k=3) → list of dicts
    * flow_metadata: list of dicts with keys: timestamp, src_ip, dst_ip,
      src_port, dst_port, protocol, stage, raw_features
    * Return top-k most attended flows with their full metadata
    * Format for UI: {rank, timestamp, src_ip, dst_port, protocol,
                      stage, attention_score, feature_contributions}

shap_explainer.py:
  - Class: FlowSHAPExplainer
    - __init__(self, model, background_data)
      * background_data: small sample of training sequences (50-100)
      * Use shap.DeepExplainer (works with PyTorch)
    - explain(self, x) → shap_values shape (T, n_features)
      * Returns SHAP values for each feature in each timestep
    - get_top_features(self, shap_values, feature_names, top_k=5) → list of
      (feature_name, mean_abs_shap) sorted descending
      * These are the flow features that most drove the forecast
    - plot_shap_summary(self, shap_values, feature_names, save_path)

Create a unified explain(model, x, flow_metadata, method='attention') function
that returns a standardised ExplanationResult:
  {
    "top_flows": [...],        # top-3 causative flow dicts
    "top_features": [...],     # top-5 causative feature names + scores
    "method": "attention",
    "confidence": 0.87         # model's confidence in the forecast
  }

Test by:
1. Load the best model + a sample attack campaign
2. Run the model forward at the Recon stage
3. Print the full ExplanationResult
4. Verify top_flows make sense (should be the flows with unusual port scan behaviour)
```

**✅ Checkpoint:** `explain()` function returns a clean dict with top flows and features. Top flows should intuitively correspond to the most suspicious-looking flows in the window.

---

## Phase 6 — FastAPI Backend

**Paste this into Claude Code:**

```
Build the FastAPI backend in sih26153/src/api/.

schemas.py — Pydantic models:
  - FlowFeatures: all 20 feature fields as float, plus metadata (src_ip, dst_ip, timestamp, etc.)
  - ForecastRequest: list of FlowFeatures (the window), model_type: str
  - ForecastResponse:
    {
      predicted_stage: str,
      stage_probabilities: dict[str, float],
      mitre_tactic_id: str,
      mitre_tactic_name: str,
      confidence: float,
      forecast_lead_flows: int,  # how many flows ahead we're forecasting
      explanation: ExplanationResult,
      model_used: str
    }
  - CampaignReplayRequest: campaign_id str, speed_multiplier float (1.0=real time)
  - ReplayEvent: flow + forecast at each timestep

replay.py — Campaign replay engine:
  - load_campaign(campaign_id) → list of (flow, stage) ordered by time
    * Load from data/processed/campaigns.json (we'll pre-compute this)
  - ReplaySession class:
    - __init__(self, campaign_id, model, explainer)
    - step() → ReplayEvent (advance one flow, run inference, return forecast)
    - reset()
    - is_complete() → bool
    - Yields events via an async generator for SSE streaming

main.py — FastAPI app:
  - Startup: load best model (auto-detect transformer or lstm), load scaler,
    init explainer, pre-load all replay campaigns
  - POST /forecast — takes ForecastRequest, returns ForecastResponse
  - GET /campaigns — list available replay campaigns with metadata
  - GET /replay/{campaign_id} — Server-Sent Events stream
    * Streams ReplayEvent as JSON every 200ms (adjustable)
    * Each event includes the full ForecastResponse for that flow
    * The frontend listens to this to drive the live demo
  - GET /health — returns model name, accuracy, dataset info
  - GET /mitre — returns full MITRE ATT&CK mapping used

Also write a script api/precompute_campaigns.py that:
  - Loads data/sequences/ 
  - Reconstructs full multi-stage campaigns (not just windows)
  - Saves top 10 most "interesting" campaigns (longest, most stages) to
    data/processed/campaigns.json
  - "Interesting" = most distinct stage transitions in the campaign

Run the API: uvicorn src.api.main:app --reload --port 8000
Test: curl http://localhost:8000/health and show the response.
Test: curl http://localhost:8000/campaigns and show campaign list.
```

**✅ Checkpoint:** FastAPI running on port 8000. `/health` returns model info. `/campaigns` returns at least 5 multi-stage campaigns. SSE stream from `/replay/{id}` emits events correctly (test with `curl -N`).

---

## Phase 7 — React Dashboard

**Paste this into Claude Code:**

```
Build the React dashboard in sih26153/dashboard/.

Initialise: npx create-react-app dashboard --template cra-template
Then: npm install recharts tailwindcss axios

The dashboard has ONE purpose: make the demo moment hit hard.
Layout: split-screen, dark theme (#0f1117 background), military/cyber aesthetic.

Components to build:

1. App.jsx — main layout
   - Left panel (60%): LiveFeed + ExplainPanel
   - Right panel (40%): ForecastPanel + MitreMap
   - Top bar: campaign selector, play/pause, speed control, model indicator

2. components/LiveFeed.jsx
   - Scrolling table of flows arriving in real-time (from SSE stream)
   - Columns: Time, Src IP, Dst Port, Protocol, Current Stage
   - Rows colour-coded by stage (Recon=blue, Exploit=orange, C2=red, etc.)
   - Most recently arrived flows glow briefly on entry (CSS transition)
   - "Causative flows" (from explanation) get a ⚡ badge and bright border

3. components/ForecastPanel.jsx
   - Large text showing: "PREDICTED NEXT STAGE: [EXPLOIT]" in the stage's colour
   - Horizontal probability bars for all 6 stages (recharts BarChart)
   - Confidence meter: circular gauge 0–100%
   - Lead-time indicator: "Forecasting X flows ahead"
   - The moment confidence > 70% for a non-Benign stage: panel flashes, plays a subtle alert

4. components/MitreMap.jsx
   - Shows current stage + predicted next stage as ATT&CK tactic cards
   - Each card: tactic ID (TA00XX), tactic name, icon, description
   - Arrow between current → predicted
   - Subsection: "Technique likely used: [technique name from MITRE]"

5. components/ExplainPanel.jsx
   - "Evidence" section: top-3 causative flows listed with their attention scores
   - Progress bars showing relative importance of each flow
   - "Key features" section: top-5 feature names with their SHAP contribution bars
   - All data comes from the explanation field in ForecastResponse

6. api/client.js
   - startReplay(campaignId, onEvent, onComplete) → EventSource connection
   - getForecast(flows) → POST to /forecast
   - getCampaigns() → GET /campaigns
   - stopReplay(eventSource)

Style requirements:
- Dark background: #0f1117
- Stage colours: Recon=#3b82f6, CredAccess=#8b5cf6, Exploit=#f97316, LateralMove=#eab308, C2=#ef4444, Impact=#dc2626, Benign=#22c55e
- Font: JetBrains Mono for IPs/ports/data, Inter for labels
- All forecast updates animate smoothly (transitions on bar widths, confidence gauge)
- Mobile-responsive: stacks vertically on small screens

Run: npm start (port 3000)
Verify: dashboard loads, campaign selector populated, play button starts SSE stream,
flows appear in LiveFeed, ForecastPanel updates with each new flow.
```

**✅ Checkpoint:** Dashboard running on port 3000. SSE stream drives live flow updates. ForecastPanel updates in real time. ExplainPanel shows top flows and features after each forecast.

---

## Phase 8 — Evaluation & Metrics

**Paste this into Claude Code:**

```
Build the evaluation suite in sih26153/src/models/evaluate.py and notebooks/02_sequence_analysis.ipynb.

evaluate.py — comprehensive evaluation:

  - load_best_model() → model + metadata
  - run_full_evaluation(model, test_sequences) → EvaluationReport:
    {
      "overall_accuracy": float,
      "top2_accuracy": float,
      "markov_accuracy": float,       # for direct comparison
      "improvement_over_markov": str, # e.g. "+18.3%"
      "mean_forecast_lead_time": float,  # in flows (THE KEY METRIC)
      "std_forecast_lead_time": float,
      "early_warning_rate": float,    # fraction of campaigns where we gave ≥2 flow warning
      "false_alarm_rate": float,      # fraction of high-confidence wrong forecasts
      "per_stage_metrics": {
          stage: {"precision": f, "recall": f, "f1": f, "support": n}
      },
      "stage_transition_accuracy": {
          "Recon→Exploit": float,     # accuracy on this specific transition
          "Recon→CredAccess": float,
          ... etc
      }
    }

  - generate_report(report, save_path) → saves as metrics.json + metrics_summary.txt
    The summary.txt should be copy-pasteable into the SIH submission form.

  - plot_lead_time_distribution(report, save_path)
    * Histogram of forecast_lead_time across all campaigns
    * Mark the mean with a vertical line
    * Title: "Flows of warning before attack progression"

  - plot_roc_per_stage(model, test_sequences, save_path)
    * One-vs-rest ROC curve for each stage
    * AUC values in legend

notebooks/02_sequence_analysis.ipynb:
  - Cell 1: Load all campaigns, show stage transition frequency as Sankey diagram
    (use matplotlib for a simple version — source stage → target stage, line thickness = frequency)
  - Cell 2: For 3 example campaigns, plot:
    * Top: flow stage labels over time (coloured timeline)
    * Bottom: model's probability for correct next stage over time
    * Mark the "correct forecast point" where prob > 0.5 for correct stage
    * Mark the "actual transition point" where the real stage changed
    * Gap between them = forecast lead time
  - Cell 3: Show EvaluationReport as formatted table
  - Cell 4: Honest limitations section — what the model gets wrong and why

Run evaluate.py and show the full EvaluationReport.
```

**✅ Checkpoint:** `metrics.json` generated. `mean_forecast_lead_time` ≥ 2.0 flows. `improvement_over_markov` is positive. Lead time distribution plot saved.

---

## Phase 9 — Demo Hardening

**Paste this into Claude Code:**

```
Prepare the project for the SIH demo presentation. This phase is about polish and reliability.

1. Select and hardcode 3 "hero campaigns" for the demo:
   - Best campaign: highest mean forecast lead time + clearest stage transitions
   - Middle campaign: shows a near-miss (model unsure, then becomes confident)
   - Hard campaign: attacker tries to blend in — model catches it late but still early
   Save these 3 as data/demo/campaign_1.json, campaign_2.json, campaign_3.json

2. Add to main.py:
   - GET /demo/campaigns — returns only these 3 campaigns
   - POST /demo/reset/{campaign_id} — resets replay to start
   - GET /demo/replay/{campaign_id}?speed=2.0 — SSE stream at 2× speed (better for demo)

3. Add to dashboard App.jsx:
   - "Demo Mode" button that loads the 3 hero campaigns in a carousel
   - Keyboard shortcut: Space=play/pause, R=reset, 1/2/3=switch campaign
   - A "SIH Judge View" button that hides the controls and maximises the forecast panel

4. Create a pre-demo health check script: scripts/demo_check.py
   - Verifies model loads correctly
   - Verifies all 3 demo campaigns load
   - Runs a single forecast and checks response structure
   - Checks API is running on port 8000
   - Checks dashboard is reachable on port 3000
   - Prints: "✅ Demo ready" or lists what's broken

5. Update README.md with:
   - "30-second demo script" section: exact words to say at each stage of the replay
   - "Judge Q&A" section with 10 anticipated questions + concise honest answers
   - "Known limitations" section (required for credibility with NTRO)

6. Add docker-compose.yml that starts both API + dashboard with one command:
   docker-compose up
   
Run demo_check.py and show the output.
```

**✅ Checkpoint:** `demo_check.py` prints all green. Three hero campaigns loaded. Demo mode keyboard shortcuts work in dashboard.

---

## Phase 10 — Tests & Final Polish

**Paste this into Claude Code:**

```
Write tests and final polish for SIH26153.

tests/test_pipeline.py:
  - test_stage_mapping_complete: all CIC-IDS labels are in STAGE_MAP
  - test_no_infinities: processed features contain no inf values
  - test_sequence_shape: X.npy is shape (N, WINDOW_SIZE, n_features), y.npy is (N,)
  - test_campaign_reconstruction: reconstructed campaigns have 2+ distinct stages

tests/test_models.py:
  - test_lstm_forward: model forward pass returns (logits, attention) with correct shapes
  - test_transformer_forward: same for transformer
  - test_model_beats_random: loaded model accuracy > 1/num_stages + 0.1
  - test_explain_output_structure: explain() returns dict with required keys

tests/test_api.py:
  - test_health_endpoint: returns 200, model name present
  - test_campaigns_endpoint: returns list of at least 3 campaigns
  - test_forecast_endpoint: POST with valid FlowFeatures returns ForecastResponse
  - test_replay_stream: SSE stream emits at least 1 event within 2 seconds

Run all tests: pytest tests/ -v
All must pass.

Final polish:
- Add a project banner to README.md showing the key metrics table
- Ensure all notebooks have "Run All" tested and passing
- Check that docker-compose up brings up both services within 30 seconds
- Add a SUBMISSION.md with: problem statement, approach summary, innovation claim,
  dataset used, evaluation metrics, team name placeholder
```

**✅ Final Checkpoint:**
- `pytest tests/ -v` → all green
- `docker-compose up` → both services running
- Dashboard plays through hero campaign without errors
- At Recon stage, model predicts Exploit/CredAccess ≥ 2 flows early
- MITRE tactic displayed, causative flows highlighted

---

## Key Numbers to Aim For

| Metric | Minimum to Demo | Good | Excellent |
|---|---|---|---|
| Test accuracy | > 50% | > 65% | > 75% |
| vs Markov improvement | > +10% | > +20% | > +30% |
| Mean forecast lead time | ≥ 2 flows | ≥ 4 flows | ≥ 6 flows |
| Early warning rate | > 50% | > 70% | > 85% |
| False alarm rate | < 30% | < 20% | < 10% |

---

## Judge Q&A — Prepare These Answers

**Q: How is this different from a normal IDS?**
A: An IDS classifies each flow after it arrives. Our model reads a sequence of flows and predicts what the attacker will do next — before it happens. We forecast; we don't just detect.

**Q: Does it work on real traffic?**
A: It's trained on research datasets (CIC-IDS2017, UNSW-NB15). For production, it would need retraining on labelled enterprise telemetry. The architecture is designed for that extension.

**Q: What if the attacker randomises their pattern?**
A: The model operates on low-level flow features (packet sizes, inter-arrival times, flag patterns), not just stage labels — so pattern randomisation has limited effect on feature-level signatures.

**Q: What does "correct forecast" mean exactly?**
A: A forecast is correct when the model assigns >60% probability to the actual next stage at least 2 flows before that stage's first flow appears in the traffic.

**Q: Why MITRE ATT&CK?**
A: It's the standard taxonomy defenders use. Mapping predictions to ATT&CK means analysts can immediately cross-reference with their existing detection rules and playbooks.

---

*Plan version: 1.0 | SIH 2026 | Problem Statement SIH26153 | NTRO*
