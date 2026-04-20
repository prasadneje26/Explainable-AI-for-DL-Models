import React, { useState, useRef } from 'react';
import axios from 'axios';

const SIGNAL_INFO = {
  sine:   { label: 'Sine Wave',   desc: 'Smooth periodic oscillation — pure tone, continuous amplitude.' },
  square: { label: 'Square Wave', desc: 'Abrupt ±1 transitions at regular intervals — digital/binary pattern.' },
  noise:  { label: 'Noise',       desc: 'Random Gaussian fluctuations — no periodicity or structure.' },
};

export default function AudioModel({ api, onResult, onError, onLoading, loading }) {
  const [mode,        setMode]        = useState('generate');
  const [signalType,  setSignalType]  = useState('sine');
  const [file,        setFile]        = useState(null);
  const ref = useRef();

  const handleSubmit = async () => {
    onLoading(true);
    try {
      const fd = new FormData();
      if (mode === 'upload' && file) fd.append('file', file);
      fd.append('signal_type', signalType);

      const res = await axios.post(`${api}/predict/audio`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } });
      if (!res.data.success) throw new Error(res.data.error);
      onResult(res.data);
    } catch (e) {
      onError(e.response?.data?.detail || e.response?.data?.error || e.message);
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="model-form">
      <h2>🎵 Audio 1D-CNN</h2>
      <p className="subtitle">
        Classify a 1D signal as Sine Wave, Square Wave, or Noise. SHAP GradientExplainer
        (Integrated Gradients) highlights which time steps drove the prediction.
      </p>

      <div className="form-group">
        <label>Input Mode</label>
        <select value={mode} onChange={e => setMode(e.target.value)}>
          <option value="generate">Generate a test signal</option>
          <option value="upload">Upload a .npy file</option>
        </select>
      </div>

      {mode === 'generate' && (
        <>
          <div className="form-group">
            <label>Signal Type</label>
            <select value={signalType} onChange={e => setSignalType(e.target.value)}>
              {Object.entries(SIGNAL_INFO).map(([key, info]) => (
                <option key={key} value={key}>{info.label}</option>
              ))}
            </select>
          </div>
          <div className="info-note">
            {SIGNAL_INFO[signalType].desc}
          </div>
        </>
      )}

      {mode === 'upload' && (
        <div
          className="file-drop"
          onClick={() => ref.current?.click()}
        >
          <input
            ref={ref} type="file" accept=".npy" style={{ display: 'none' }}
            onChange={e => e.target.files[0] && setFile(e.target.files[0])}
          />
          <span className="drop-icon">📂</span>
          <p>{file ? file.name : 'Click to upload a .npy signal file'}</p>
          <small>NumPy array of float32 values (any length, resampled to 500 samples)</small>
        </div>
      )}

      <button
        className="btn-primary"
        onClick={handleSubmit}
        disabled={(mode === 'upload' && !file) || loading}
        style={{ marginTop: '1.25rem' }}
      >
        {loading ? 'Analyzing…' : 'Predict & Explain'}
      </button>

      <div className="info-note" style={{ marginTop: '0.75rem' }}>
        <strong>Classes:</strong> Sine Wave · Square Wave · Noise.
        The 1D-CNN was trained on 800 synthetic signals per class.
        SHAP GradientExplainer is fast (~5–10 s).
      </div>
    </div>
  );
}
