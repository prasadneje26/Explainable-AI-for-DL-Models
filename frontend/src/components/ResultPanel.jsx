import React from 'react';
import './ResultPanel.css';

export default function ResultPanel({ result }) {
  if (!result) return null;

  const conf = (result.confidence * 100).toFixed(1);

  return (
    <div className="result-panel">
      {/* Header */}
      <div className="result-header">
        <div className="result-model-badge">{result.model}</div>
        <div className="result-input-type">{result.input_type} · {result.dataset}</div>
      </div>

      {/* Prediction */}
      <div className="result-prediction">
        <span className="pred-label">{result.prediction}</span>
        <span className="pred-conf">{conf}% confidence</span>
        {conf < 50 && <span className="low-conf-warning">⚠️ Low confidence prediction</span>}
      </div>

      {/* Probability bars */}
      <div className="prob-bars">
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

      {/* SHAP explanation */}
      <div className="shap-section">
        <h4>SHAP Explanation</h4>
        <p className="shap-desc">{result.explanation}</p>
        <p className="shap-hint">{result.shap_description}</p>

        {/* Image overlay */}
        {result.shap_overlay && (
          <div className="shap-images">
            <div className="shap-img-wrap">
              <img src={result.shap_overlay} alt="SHAP overlay" />
              <p>SHAP Overlay</p>
            </div>
          </div>
        )}

        {/* SHAP plot (all models) */}
        {result.shap_plot && (
          <div className="shap-plot-wrap">
            <img src={result.shap_plot} alt="SHAP plot" />
          </div>
        )}
      </div>
    </div>
  );
}
