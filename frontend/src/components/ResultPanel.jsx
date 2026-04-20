import React, { useState } from 'react';
import './ResultPanel.css';

export default function ResultPanel({ result }) {
  const [showTheory, setShowTheory] = useState(false);
  if (!result) return null;

  const conf = (result.confidence * 100).toFixed(1);
  const confColor = result.confidence > 0.9 ? '#10b981' :
                    result.confidence > 0.7 ? '#f59e0b' : '#ef4444';

  return (
    <div className="result-panel">

      {/* Model badge */}
      <div className="result-header">
        <span className="model-badge">{result.model}</span>
        <span className="dataset-badge">{result.dataset}</span>
        <span className="method-badge">SHAP {result.shap_method}</span>
      </div>

      {/* Main prediction */}
      <div className="prediction-box">
        <div className="pred-main">
          <span className="pred-label">{result.prediction}</span>
          <span className="pred-conf" style={{color: confColor}}>{conf}%</span>
        </div>
        <p className="pred-sub">{result.input_type}</p>
      </div>

      {/* Probability bars */}
      <div className="prob-section">
        <h4>Class Probabilities</h4>
        {result.all_predictions?.map((p, i) => (
          <div key={i} className="prob-row">
            <span className="prob-name">{p.class}</span>
            <div className="prob-track">
              <div className={`prob-fill ${p.class === result.prediction ? 'top' : ''}`}
                style={{ width: `${(p.probability * 100).toFixed(1)}%` }} />
            </div>
            <span className="prob-val">{(p.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      {/* SHAP visuals */}
      <div className="shap-section">
        <h4>SHAP Explanation — {result.shap_method}</h4>
        <p className="shap-hint">{result.shap_description}</p>

        {result.shap_overlay && (
          <div className="overlay-wrap">
            <img src={result.shap_overlay} alt="SHAP overlay" />
            <p>Pixel-level SHAP overlay</p>
          </div>
        )}

        {result.shap_plot && (
          <div className="plot-wrap">
            <img src={result.shap_plot} alt="SHAP plot" />
          </div>
        )}
      </div>

      {/* Deep explanation */}
      {result.deep_explanation && (
        <div className="deep-section">
          <h4>What SHAP Found</h4>
          <p>{result.deep_explanation}</p>
        </div>
      )}

      {/* Model theory toggle */}
      {result.theory && (
        <div className="theory-section">
          <button className="theory-toggle" onClick={() => setShowTheory(!showTheory)}>
            {showTheory ? '▲ Hide' : '▼ Show'} Model Theory
          </button>
          {showTheory && (
            <div className="theory-body">
              <p><strong>Architecture:</strong> {result.architecture}</p>
              <p>{result.theory}</p>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
