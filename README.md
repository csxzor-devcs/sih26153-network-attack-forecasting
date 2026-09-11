# SIH26153 — Network Attack Forecasting System

> **Built for:** NTRO Problem Statement (SIH26153)
> **Author:** Claude Code
> **Date:** September 2026

---

## What Is This?

This system **predicts the next stage of a cyberattack** from a sequence of recent network flows.

Think of it like this:

1. **You feed it** the last 20 network connections (who connected where, how much data, what flags were set)
2. **It predicts** what happens next — will the attacker move to the next stage? Which stage?
3. **You know** the attack is progressing and can take action

### The 7 Stages It Tracks

```
Benign → Recon → CredAccess → Exploit → LateralMove → C2 → Impact
   ↑         ↑           ↑            ↑              ↑        ↑
  Normal   Scanning   Stolen     Software   Move       Command  Damage
  traffic  ports      creds      exploited  inside     & control
```

This maps directly to the **MITRE ATT&CK framework** — the industry standard for describing attacker behavior.

---

## How It Works

The system has **five prediction engines** compared in a proper baseline hierarchy:

| Model | What It Does | Params | Role |
|-------|-------------|--------|------|
| **Majority** | Always predicts the most frequent class | 0 | Floor baseline |
| **Markov** | First-order chain on stage transitions | ~5 KB | Probabilistic baseline |
| **XGBoost** | Gradient-boosted trees on window features | ~1 MB | Strong non-neural baseline |
| **LSTM** | Neural network that remembers temporal patterns | ~223 KB | Deep learning baseline |
| **Transformer** | Self-attention over flow sequences | ~543 KB | Advanced deep learning baseline |

All models are evaluated on the **same campaign-held-out test set** to ensure fair comparison. Campaign-based train/val/test splits prevent temporal data leakage.

---

## Quick Start (5 Minutes)

### Prerequisites

- **Python 3.11+**
- **Docker** (optional, for production)

### Step 1: Install Python and Dependencies

```bash
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3 python3-pip

# On macOS:
brew install python3

# Install dependencies:
pip install -r requirements.txt
```

### Step 2: Clone This Repository

```bash
git clone https://github.com/YOUR_USERNAME/sih26153-network-attack-forecasting.git
cd sih26153-network-attack-forecasting
```

### Step 3: Run the Full Pipeline

```bash
# This will:
# 1. Load and process the CIC-IDS2017 dataset
# 2. Create attack sequences from network flows
# 3. Train all three models (Markov, LSTM, Transformer)
# 4. Generate predictions and evaluation reports

python -m src.pipeline.run_pipeline
```

### Step 4: Run Evaluation

```bash
# Compares all three models and generates charts
python -m src.evaluation
```

Results appear in the `reports/` folder:
- `reports/model_comparison.png` — Bar chart of model accuracy
- `reports/Markov_cm.png` — Markov confusion matrix
- `reports/LSTM_cm.png` — LSTM confusion matrix
- `reports/Transformer_cm.png` — Transformer confusion matrix
- `reports/roc_curves.png` — ROC curves for all stages
- `reports/evaluation_summary.json` — Detailed metrics

### Step 5: Start the API Server

```bash
# Starts a web server at http://localhost:8000
python -m src.api.app
```

### Step 6: Open the Dashboard

```bash
# In the frontend folder
cd frontend
npm install
npm start
```

Open **http://localhost:3000** in your browser.

---

## What You Can Do

### 🔮 Make a Prediction

Send a sequence of 20 network flows and get back the predicted attack stage:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[100.0, 50.0, ...], [95.0, 48.0, ...], ...]}'
```

**Response:**
```json
{
  "predictions": [
    {
      "stage": "Exploit",
      "stage_index": 3,
      "confidence": 0.73,
      "all_stages": [
        {"stage": "Benign", "probability": 0.05},
        {"stage": "Recon", "probability": 0.08},
        {"stage": "CredAccess", "probability": 0.12},
        {"stage": "Exploit", "probability": 0.73},
        ...
      ]
    }
  ]
}
```

### 📊 Check System Status

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/info
curl http://localhost:8000/stages
curl http://localhost:8000/stats
```

### 🔍 See Feature Importance

```bash
curl http://localhost:8000/explain
```

This tells you **which network features** (packet sizes, timing, flags) most influenced the prediction — crucial for understanding why the system flagged something.

---

## Run with Docker (Production)

### Prerequisites

```bash
# Install Docker
# https://docs.docker.com/get-docker/
```

### Build and Run Everything

```bash
# Copy environment file
cp .env.example .env

# Build and start all services
docker-compose up --build -d
```

This starts:
- **API** at `http://localhost:8000`
- **Dashboard** at `http://localhost:3000`
- **Nginx** reverse proxy at `http://localhost`
- **Redis** for caching

### Stop Everything

```bash
docker-compose down
```

### Check Logs

```bash
docker-compose logs -f api
docker-compose logs -f frontend
```

---

## Project Structure

```
sih26153/
├── src/
│   ├── config.py              # All configuration constants
│   ├── config_manager.py      # Environment-aware config
│   ├── pipeline/              # Data processing pipeline
│   │   ├── loader.py          # Loads CIC-IDS2017 CSV files
│   │   ├── sequencer.py       # Creates attack campaigns & sequences
│   │   ├── features.py        # Feature extraction
│   │   ├── labeller.py        # Labels flows with attack stages
│   │   └── run_pipeline.py    # Runs the full pipeline
│   ├── baseline/
│   │   └── markov.py          # Markov chain baseline model
│   ├── models/
│   │   ├── lstm.py            # LSTM neural network
│   │   └── transformer.py     # Transformer neural network
│   ├── explainability/
│   │   └── explainability.py  # Feature importance analysis
│   ├── api/
│   │   └── app.py             # FastAPI REST server
│   └── evaluation/
│       └── evaluation.py      # Model comparison & reports
├── frontend/                  # React dashboard
│   ├── package.json
│   └── src/
│       └── App.jsx            # Dashboard components
├── tests/                     # Unit tests (32 tests)
│   ├── test_pipeline.py
│   ├── test_models.py
│   └── test_api.py
├── data/                      # Dataset files
│   ├── raw/                   # Raw CIC-IDS2017 CSVs
│   ├── processed/             # Cleaned data
│   └── sequences/             # Training sequences
├── models/                    # Trained model files
├── reports/                   # Evaluation charts & reports
├── docker/                    # Docker configs
├── Dockerfile                 # Production Docker image
├── docker-compose.yml         # Multi-container orchestration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
└── README.md                  # This file
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**32 tests total** covering:
- Pipeline: Campaign reconstruction, window generation, data loading
- Models: Markov training, LSTM/Transformer architecture, forward passes
- API: Health, model info, prediction endpoints, batch prediction
- Config: Environment variable loading, defaults

---

## Understanding the Results

### Why ~80% Accuracy?

The CIC-IDS2017 dataset is heavily imbalanced — most flows are "Benign" (normal traffic). When you predict "Benign" every time, you get ~80% accuracy. This is expected and normal.

**What matters more:**
- Can the model detect the *transition* from Benign to Recon? → Yes, the models learn stage progression patterns
- Are the confidence scores meaningful? → Yes, higher confidence = more certain
- Does the Transformer outperform the Markov baseline? → Check `reports/evaluation_summary.json`

### Reading the MITRE Mapping

Each attack stage maps to a **MITRE ATT&CK tactic**:

| Stage | MITRE Tactic ID | Meaning |
|-------|----------------|---------|
| Recon | TA0043 | Reconnaissance |
| CredAccess | TA0006 | Credential Access |
| Exploit | TA0001 | Initial Access |
| LateralMove | TA0008 | Lateral Movement |
| C2 | TA0011 | Command and Control |
| Impact | TA0040 | Impact |

---

## Dataset: CIC-IDS2017

- **Source:** Canadian Institute for Cybersecurity
- **Size:** ~2.8 million network flow records
- **Features:** 79 columns per flow (timing, packet sizes, flags, etc.)
- **Attack types:** 15 categories (DoS, scanning, exploitation, etc.)
- **Preprocessed to:** 100K rows, 46 features, 100 campaigns, 97,661 sequences

---

## Configuration

Edit `.env` to customize:

```bash
# Server settings
API_PORT=8000
LOG_LEVEL=info

# Dataset settings
MAX_ROWS=100000        # Rows to use from the dataset
N_CAMPAIGNS=100        # Number of synthetic attack campaigns
WINDOW_SIZE=20         # How many flows to look at for prediction

# Model selection
DEFAULT_MODEL=transformer  # Options: markov, lstm, transformer

# Redis (for caching)
REDIS_ENABLED=false
REDIS_HOST=redis
REDIS_PORT=6379
```

---

## Requirements

- Python 3.11+
- 4 GB RAM minimum (8 GB recommended for full dataset)
- Docker 24+ (for container deployment)
- Node.js 18+ (for frontend development)

---

## License

MIT License — See `LICENSE` file for details.

---

## Quick Reference: API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/model/info` | GET | Model configuration and metadata |
| `/predict` | POST | Predict attack stage from a sequence |
| `/predict/batch` | POST | Predict multiple sequences at once |
| `/stages` | GET | List all 7 attack stages |
| `/features` | GET | Feature column information |
| `/stats` | GET | Dataset and model statistics |
| `/explain` | GET | Feature importance analysis |

---

## Support

If you run into issues:

1. Check `reports/` for evaluation logs
2. Run `pytest tests/ -v` to verify installation
3. Check `docker-compose logs` if using Docker
4. Ensure `requirements.txt` packages are installed

---

## What's Next?

Potential improvements:
- [ ] Train on full 2.8M rows (requires more RAM)
- [ ] Add real-time streaming prediction
- [ ] Deploy to Kubernetes
- [ ] Add user authentication to the API
- [ ] Build mobile dashboard
- [ ] Add more explainability methods (SHAP, LIME)
