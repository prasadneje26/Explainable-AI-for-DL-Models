import React from 'react';
import './ExplanationView.css';

function ExplanationView({ explanations }) {
  if (!explanations) return null;

  return (
    <div className="explanation-view">
      <h2>Explainable AI Explanations</h2>

      {Object.entries(explanations).map(([method, data]) => (
        <div key={method} className="explanation-card">
          <div className="explanation-header">
            <h3>{method.toUpperCase()}</h3>
            <span className="badge">{method === 'shap' ? 'SHapley Additive exPlanations' : method === 'lime' ? 'Local Interpretable Model-agnostic Explanations' : method === 'gradcam' ? 'Gradient-weighted Class Activation Mapping' : ''}</span>
          </div>

          <div className="explanation-content">
            <p className="description">{data.description}</p>

            <div className="images-row">
              {data.overlay && (
                <div className="explanation-image">
                  <img src={data.overlay} alt={`${method} Overlay`} />
                  <p className="image-caption">{method} overlay</p>
                </div>
              )}
              {data.plot && (
                <div className="explanation-image">
                  <img src={data.plot} alt={`${method} Plot`} />
                  <p className="image-caption">{method} plot</p>
                </div>
              )}
              {data.heatmap && (
                <div className="explanation-image">
                  <img src={data.heatmap} alt={`${method} Heatmap`} />
                  <p className="image-caption">{method} heatmap</p>
                </div>
              )}
            </div>

            {data.explanation && (
              <div className="lime-explanation">
                <h4>LIME Word Importance:</h4>
                <ul>
                  {data.explanation.map(([word, score], i) => (
                    <li key={i} style={{ color: score > 0 ? 'green' : 'red' }}>
                      {word}: {score.toFixed(3)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      ))}

      {/* Comparison */}
      <div className="comparison-box">
        <h3>XAI Methods Comparison</h3>
        <div className="comparison-table">
          <div className="comparison-cell">
            <strong>SHAP</strong>
            <p>Game theory (Shapley values)</p>
            <p>Pixel-level attribution</p>
            <p>Model-agnostic</p>
            <p>Theoretically grounded</p>
          </div>
          <div className="comparison-cell">
            <strong>Grad-CAM</strong>
            <p>Gradient-based</p>
            <p>Coarse spatial map</p>
            <p>CNN-specific</p>
            <p>Fast</p>
          </div>
          <div className="comparison-cell">
            <strong>LIME</strong>
            <p>Perturbation-based</p>
            <p>Superpixel regions</p>
            <p>Model-agnostic</p>
            <p>Approximate</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ExplanationView;
