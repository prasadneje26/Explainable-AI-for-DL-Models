import React from 'react';
import './PredictionResult.css';

function PredictionResult({ prediction }) {
  if (!prediction || !prediction.success) return null;

  return (
    <div className="prediction-result">
      <div className="result-header">
        <h2>🎯 Prediction Result</h2>
        <div className="main-prediction">
          <span className="class-name">{prediction.prediction}</span>
          <span className="confidence">{(prediction.confidence * 100).toFixed(1)}% confidence</span>
        </div>
      </div>

      <div className="probabilities-chart">
        <h3>📊 Class Probabilities</h3>
        {prediction.all_predictions.map((pred, index) => (
          <div key={index} className="prob-row">
            <span className="prob-label">{pred.class}</span>
            <div className="prob-bar-container">
              <div 
                className={`prob-bar ${pred.class === prediction.prediction ? 'active' : ''}`}
                style={{ width: `${pred.probability * 100}%` }}
              ></div>
            </div>
            <span className="prob-value">{(pred.probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PredictionResult;
