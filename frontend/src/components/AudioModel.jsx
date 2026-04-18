import React, { useState, useRef } from 'react';
import axios from 'axios';

export default function AudioModel({ api, onResult, onError, onLoading, loading }) {
  const [mode,       setMode]       = useState('generate'); // 'generate' | 'upload'
  const [signalType, setSignalType] = useState('sine');
  const [file,       setFile]       = useState(null);
  const ref = useRef();

  const handleSubmit = async () => {
    onLoading(true);
    try {
      const fd = new FormData();
      if (mode === 'upload' && file) {
        fd.append('file', file);
      }
      fd.append('signal_type', signalType);

      const res = await axios.post(`${api}/predict/audio`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } });
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
      <h2>🎵 Audio 1D-CNN</h2>
      <p style={{color:'#64748b',fontSize:'0.875rem',marginTop:0}}>
        Classify a 1D signal as Sine Wave, Square Wave, or Noise
      </p>

      <div className="form-group">
        <label>Input Mode</label>
        <select value={mode} onChange={e => setMode(e.target.value)}>
          <option value="generate">Generate test signal</option>
          <option value="upload">Upload .npy file</option>
        </select>
      </div>

      {mode === 'generate' && (
        <div className="form-group">
          <label>Signal Type to Generate</label>
          <select value={signalType} onChange={e => setSignalType(e.target.value)}>
            <option value="sine">Sine Wave</option>
            <option value="square">Square Wave</option>
            <option value="noise">Noise</option>
          </select>
        </div>
      )}

      {mode === 'upload' && (
        <div className="file-drop" onClick={() => ref.current?.click()}>
          <input ref={ref} type="file" accept=".npy" style={{display:'none'}}
            onChange={e => e.target.files[0] && setFile(e.target.files[0])} />
          <p style={{fontSize:'1.5rem',margin:0}}>📂</p>
          <p>{file ? file.name : 'Click to upload .npy signal file'}</p>
        </div>
      )}

      <div style={{background:'#f0fdf4',border:'1px solid #86efac',borderRadius:'8px',
        padding:'0.75rem',marginTop:'1rem',fontSize:'0.8rem',color:'#166534'}}>
        <strong>Signal Classes:</strong> Sine Wave (pure tone) · Square Wave (digital) · Noise (random)
      </div>

      <button className="btn-primary" onClick={handleSubmit}
        disabled={(mode === 'upload' && !file) || loading}>
        {loading ? 'Analyzing...' : 'Predict & Explain'}
      </button>
    </div>
  );
}
