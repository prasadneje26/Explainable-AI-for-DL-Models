import React, { useState } from 'react';
import axios from 'axios';

const EXAMPLES = [
  { label: 'Setosa',     vals: [5.1, 3.5, 1.4, 0.2] },
  { label: 'Versicolor', vals: [5.9, 3.0, 4.2, 1.5] },
  { label: 'Virginica',  vals: [6.7, 3.1, 5.6, 2.4] },
];

const FIELDS = [
  { key: 'sepal_length', label: 'Sepal Length (cm)', min: 4.0, max: 8.0, step: 0.1, placeholder: 'e.g. 5.1' },
  { key: 'sepal_width',  label: 'Sepal Width (cm)',  min: 2.0, max: 5.0, step: 0.1, placeholder: 'e.g. 3.5' },
  { key: 'petal_length', label: 'Petal Length (cm)', min: 1.0, max: 7.0, step: 0.1, placeholder: 'e.g. 1.4' },
  { key: 'petal_width',  label: 'Petal Width (cm)',  min: 0.1, max: 3.0, step: 0.1, placeholder: 'e.g. 0.2' },
];

export default function TabularModel({ api, onResult, onError, onLoading, loading }) {
  const [vals, setVals] = useState({
    sepal_length: '', sepal_width: '', petal_length: '', petal_width: ''
  });

  const setExample = (v) => setVals({
    sepal_length: v[0], sepal_width: v[1], petal_length: v[2], petal_width: v[3]
  });

  const valid = FIELDS.every(f => vals[f.key] !== '' && !isNaN(vals[f.key]));

  const handleSubmit = async () => {
    if (!valid) return;
    onLoading(true);
    try {
      const body = Object.fromEntries(FIELDS.map(f => [f.key, parseFloat(vals[f.key])]));
      const res  = await axios.post(`${api}/predict/tabular`, body);
      if (!res.data.success) throw new Error(res.data.error);
      onResult(res.data);
    } catch (e) {
      onError(e.response?.data?.error || e.message);
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="model-form">
      <h2>📊 Tabular DNN</h2>
      <p className="subtitle">
        Enter 4 Iris flower measurements. The DNN classifies the species and SHAP KernelExplainer
        shows the exact contribution of each measurement to the decision.
      </p>

      {FIELDS.map(f => (
        <div className="form-group" key={f.key}>
          <label>{f.label}</label>
          <input
            type="number"
            step={f.step}
            min={f.min}
            max={f.max}
            placeholder={f.placeholder}
            value={vals[f.key]}
            onChange={e => setVals(p => ({ ...p, [f.key]: e.target.value }))}
          />
        </div>
      ))}

      <div className="examples-section">
        <span className="examples-label">Load a typical example</span>
        <div className="example-chips">
          {EXAMPLES.map((ex, i) => (
            <button key={i} className="chip" onClick={() => setExample(ex.vals)}>
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      <button className="btn-primary" onClick={handleSubmit} disabled={!valid || loading}>
        {loading ? 'Analyzing…' : 'Predict & Explain'}
      </button>

      <div className="info-note">
        <strong>Reference ranges:</strong> Setosa petals 1.0–1.9 cm · Versicolor 3.0–5.1 cm · Virginica 4.5–6.9 cm.
        SHAP analysis takes ~30–60 s.
      </div>
    </div>
  );
}
