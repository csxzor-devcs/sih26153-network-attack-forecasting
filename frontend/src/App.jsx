/**
 * App.jsx -- React dashboard for SIH26153 attack forecasting.
 *
 * Visualizes:
 * - Attack stage predictions and confidence
 * - Stage transition matrix
 * - Model performance comparison (Markov vs LSTM vs Transformer)
 * - Feature importance charts
 * - Real-time prediction feed
 */

import React, { useState, useEffect } from 'react';

// API base URL
const API_BASE = 'http://localhost:8000';

/**
 * Main dashboard component.
 */
function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [modelInfo, setModelInfo] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch model info on mount
  useEffect(() => {
    fetchModelInfo();
    fetchStats();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch(`${API_BASE}/model/info`);
      const data = await res.json();
      setModelInfo(data);
    } catch (e) {
      console.error('Failed to fetch model info:', e);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(`${API_BASE}/stats`);
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    }
  };

  const predictStage = async (features) => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
      });
      const data = await res.json();
      setPredictions(data.predictions);
    } catch (e) {
      console.error('Prediction failed:', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <header>
        <h1>SIH26153 Network Attack Forecasting</h1>
        <nav>
          <button onClick={() => setActiveTab('overview')}>Overview</button>
          <button onClick={() => setActiveTab('predictions')}>Predictions</button>
          <button onClick={() => setActiveTab('models')}>Models</button>
          <button onClick={() => setActiveTab('explainability')}>Explainability</button>
        </nav>
      </header>

      <main>
        {activeTab === 'overview' && (
          <OverviewSection modelInfo={modelInfo} stats={stats} loading={loading} />
        )}
        {activeTab === 'predictions' && (
          <PredictionsSection
            predictions={predictions}
            loading={loading}
            onPredict={predictStage}
          />
        )}
        {activeTab === 'models' && (
          <ModelsSection modelInfo={modelInfo} stats={stats} />
        )}
        {activeTab === 'explainability' && (
          <ExplainabilitySection />
        )}
      </main>

      <footer>
        <p>SIH26153 — Network Attack Forecasting System</p>
      </footer>
    </div>
  );
}

/** Overview section showing model info and stats. */
function OverviewSection({ modelInfo, stats, loading }) {
  return (
    <section className="overview">
      <h2>System Overview</h2>
      {loading && <p>Loading...</p>}
      {modelInfo && (
        <div className="model-card">
          <h3>{modelInfo.model_type.toUpperCase()} Model</h3>
          <p>Stages: {modelInfo.n_stages}</p>
          <p>Window Size: {modelInfo.window_size}</p>
          <p>Features: {modelInfo.n_features}</p>
          <ul>
            {modelInfo.stage_names?.map((s, i) => (
              <li key={i}>{s} (idx: {i})</li>
            ))}
          </ul>
        </div>
      )}
      {stats && (
        <div className="stats-card">
          <h3>Dataset Statistics</h3>
          <p>Sequences: {stats.dataset?.num_sequences}</p>
          <p>Shape: {stats.dataset?.sequence_shape?.join(' × ')}</p>
          <div className="stage-dist">
            <h4>Stage Distribution</h4>
            {Object.entries(stats.stage_distribution || {}).map(
              ([stage, count]) => (
                <div key={stage} className="stage-bar">
                  <span>{stage}: {count}</span>
                  <div
                    className="bar-fill"
                    style={{
                      width: `${(count / stats.dataset?.num_sequences * 100) || 0}%`,
                    }}
                  />
                </div>
              )
            )}
          </div>
        </div>
      )}
    </section>
  );
}

/** Predictions section for real-time stage forecasting. */
function PredictionsSection({ predictions, loading, onPredict }) {
  const [inputFeatures, setInputFeatures] = useState('');

  const handlePredict = () => {
    try {
      const features = JSON.parse(inputFeatures);
      if (!Array.isArray(features) || !Array.isArray(features[0])) {
        alert('Enter a valid 2D feature array (e.g., [[1.0, 2.0, ...], ...])');
        return;
      }
      onPredict(features);
    } catch (e) {
      alert('Invalid JSON format');
    }
  };

  return (
    <section className="predictions">
      <h2>Real-time Predictions</h2>
      <div className="prediction-input">
        <textarea
          placeholder="Paste feature JSON array here (window_size × n_features)"
          value={inputFeatures}
          onChange={(e) => setInputFeatures(e.target.value)}
          rows={10}
          cols={80}
        />
        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>
      {predictions.length > 0 && (
        <div className="prediction-results">
          <h3>Results</h3>
          {predictions.map((pred, i) => (
            <div key={i} className="prediction-item">
              <span className="stage">{pred.stage}</span>
              <span className="confidence">
                Confidence: {(pred.confidence * 100).toFixed(1)}%
              </span>
              <div className="stage-probs">
                {pred.all_stages?.map((s, j) => (
                  <div key={j} className="prob-bar">
                    <span>{s.stage}: {(s.probability * 100).toFixed(1)}%</span>
                    <div
                      className="prob-fill"
                      style={{ width: `${s.probability * 100}%` }}
                    />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/** Models section comparing Markov, LSTM, and Transformer. */
function ModelsSection({ modelInfo, stats }) {
  const [models, setModels] = useState([]);

  useEffect(() => {
    fetch(`${API_BASE}/model/info`)
      .then((r) => r.json())
      .then((data) => {
        setModels([
          { name: 'Markov', accuracy: 0.8015, path: 'models/markov_baseline.npz' },
          { name: 'LSTM', accuracy: 0.8015, path: 'models/lstm_best.pth' },
          { name: 'Transformer', accuracy: 0.8015, path: 'models/transformer_best.pth' },
        ]);
      })
      .catch(console.error);
  }, []);

  return (
    <section className="models">
      <h2>Model Comparison</h2>
      <table className="model-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Validation Accuracy</th>
            <th>Parameters</th>
            <th>Path</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.name}>
              <td>{m.name}</td>
              <td>{(m.accuracy * 100).toFixed(2)}%</td>
              <td>-</td>
              <td>{m.path}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        All models achieve ~80% accuracy (Benign majority class).
        Focus is on detecting anomalous transitions, not overall accuracy.
      </p>
    </section>
  );
}

/** Explainability section showing feature importance. */
function ExplainabilitySection() {
  const [featureImportance, setFeatureImportance] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/explain`)
      .then((r) => r.json())
      .then(setFeatureImportance)
      .catch(console.error);
  }, []);

  return (
    <section className="explainability">
      <h2>Feature Importance</h2>
      {featureImportance ? (
        <div className="importance-list">
          {featureImportance.top_features?.map(([feat, imp], i) => (
            <div key={i} className="importance-item">
              <span className="feat-name">{feat}</span>
              <div className="importance-bar">
                <div
                  className="importance-fill"
                  style={{ width: `${Math.min(imp * 1000, 100)}%` }}
                />
              </div>
              <span className="imp-value">{imp.toFixed(6)}</span>
            </div>
          ))}
        </div>
      ) : (
        <p>Loading feature importance...</p>
      )}
    </section>
  );
}

export default App;
