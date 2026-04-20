import React, { useState } from 'react';
import './ResultPanel.css';

export default function ResultPanel({ result }) {
  const [showTheory, setShowTheory] = useState(false);
  if (!result) return null;

  const conf     = (result.confidence * 100).toFixed(1);
  const confColor = result.confidence > 0.9 ? '#16a34a'
                  : result.confidence > 0.7 ? '#d97706'
                  : '#dc2626';

  return (
    <div className="result-panel">

      {/* Badges */}
      <div className="result-badges">
        <span className="badge badge-model">{result.model}</span>
        <span className="badge badge-dataset">{result.dataset}</span>
        <span className="badge badge-method">SHAP {result.shap_method}</span>
      </div>

      {/* Prediction */}
      <div className="prediction-box">
        <div className="pred-left">
          <div className="pred-label">{result.prediction}</div>
          <div className="pred-sub">{result.input_type}</div>
        </div>
        <div className="pred-conf-block">
          <div className="pred-conf-value" style={{ color: confColor }}>{conf}%</div>
          <div className="pred-conf-label">Confidence</div>
        </div>
      </div>

      {/* Probabilities */}
      <div className="probs-section">
        <p className="probs-title">Class Probabilities</p>
        {result.all_predictions?.map((p, i) => (
          <div key={i} className="prob-row">
            <span className="prob-name">{p.class}</span>
            <div className="prob-track">
              <div
                className={`prob-fill ${p.class === result.prediction ? 'top' : ''}`}
                style={{ width: `${(p.probability * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="prob-val">{(p.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <div className="divider" />

      {/* SHAP Visuals */}
      <div className="shap-section">
        <p className="section-title">SHAP Explanation — {result.shap_method}</p>
        <p className="shap-hint">{result.shap_description}</p>

        {result.shap_overlay && (
          <div className="overlay-row">
            <div className="overlay-box">
              <img src={result.shap_overlay} alt="SHAP pixel overlay" />
              <p>Pixel overlay</p>
            </div>
          </div>
        )}

        {result.shap_plot && (
          <img src={result.shap_plot} alt="SHAP explanation plot" className="shap-plot" />
        )}
      </div>

      <div className="divider" />

      {/* Detailed Explanation */}
      <div className="explanation-section">
        <p className="section-title">Analysis</p>

        {result.deep_explanation && (
          <div className="explanation-summary">
            {result.deep_explanation}
          </div>
        )}

        {result.explanation_bullets?.length > 0 && (
          <ul className="bullets-list">
            {result.explanation_bullets.map((item, i) => (
              <li key={i} className="bullet-item">
                <span className="bullet-icon">{i + 1}</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Model Theory (collapsible) */}
      {result.theory && (
        <div className="theory-section">
          <button className="theory-toggle" onClick={() => setShowTheory(!showTheory)}>
            <span>{showTheory ? '▲' : '▼'}</span>
            <span>Model Architecture & Theory</span>
          </button>
          {showTheory && (
            <div className="theory-body">
              <p><strong>Architecture:</strong> {result.architecture}</p>
              <p style={{ marginTop: '0.5rem' }}>{result.theory}</p>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
