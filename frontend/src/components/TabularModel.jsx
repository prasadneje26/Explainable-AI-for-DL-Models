import React, { useState } from 'react';
import axios from 'axios';

const EXAMPLES = [
  { label: 'Setosa (typical)',     vals: [5.1, 3.5, 1.4, 0.2] },
  { label: 'Versicolor (typical)', vals: [5.9, 3.0, 4.2, 1.5] },
  { label: 'Virginica (typical)',  vals: [6.7, 3.1, 5.6, 2.4] },
];

const FIELDS = [
  { key: 'sepal_length', label: 'Sepal Length (cm)', placeholder: 'e.g. 5.1' },
  { key: 'sepal_width',  label: 'Sepal Width (cm)',  placeholder: 'e.g. 3.5' },
  { key: 'petal_length', label: 'Petal Length (cm)', placeholder: 'e.g. 1.4' },
  { key: 'petal_width',  label: 'Petal Width (cm)',  placeholder: 'e.g. 0.2' },
];

export default function TabularModel({ api, onResult, onError, onLoading, loading }) {
  const [vals, setVals] = useState({ sepal_length:'', sepal_width:'', petal_length:'', petal_width:'' });

  const setExample = (v) => setVals({
    sepal_length: v[0], sepal_width: v[1], petal_length: v[2], petal_width: v[3]
  });

  const valid = FIELDS.every(f => vals[f.key] !== '' && !isNaN(vals[f.key]));

  const handleSubmit = async () => {
    if (!valid) return;
    onLoading(true);
    try {
      const body = Object.fromEntries(FIELDS.map(f => [f.key, parseFloat(vals[f.key])]));
      const res = await axios.post(`${api}/predict/tabular`, body);
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
      <p style={{color:'#64748b',fontSize:'0.875rem',marginTop:0}}>
        Enter 4 Iris flower measurements — classified into species
      </p>

      {FIELDS.map(f => (
        <div className="form-group" key={f.key}>
          <label>{f.label}</label>
          <input type="number" step="0.1" placeholder={f.placeholder}
            value={vals[f.key]}
            onChange={e => setVals(p => ({...p, [f.key]: e.target.value}))} />
        </div>
      ))}

      <div style={{marginBottom:'1rem'}}>
        <label style={{fontSize:'0.8rem',color:'#64748b',fontWeight:600}}>Load example:</label>
        <div style={{display:'flex',gap:'0.5rem',marginTop:'0.4rem',flexWrap:'wrap'}}>
          {EXAMPLES.map((ex, i) => (
            <button key={i} onClick={() => setExample(ex.vals)}
              style={{padding:'0.35rem 0.7rem',fontSize:'0.78rem',
                background:'#f1f5f9',border:'1px solid #e2e8f0',borderRadius:'6px',cursor:'pointer'}}>
              {ex.label}
            </button>
          ))}
        </div>
      </div>

      <button className="btn-primary" onClick={handleSubmit} disabled={!valid || loading}>
        {loading ? 'Analyzing...' : 'Predict & Explain'}
      </button>
    </div>
  );
}
