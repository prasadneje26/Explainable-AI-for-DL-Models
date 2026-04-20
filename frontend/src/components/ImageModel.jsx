import React, { useRef, useState } from 'react';
import axios from 'axios';

export default function ImageModel({ api, onResult, onError, onLoading, loading }) {
  const [preview, setPreview] = useState(null);
  const [file,    setFile]    = useState(null);
  const ref = useRef();

  const handleFile = (f) => {
    setFile(f);
    const reader = new FileReader();
    reader.onload = e => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  };

  const handleSubmit = async () => {
    if (!file) return;
    onLoading(true);
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await axios.post(`${api}/predict/image`, fd);
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
      <h2>🖼️ Image CNN</h2>
      <p className="subtitle">
        Upload any photo and the CIFAR-10 CNN will classify it into one of 10 categories:
        airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
        SHAP DeepExplainer then highlights the exact pixels that drove the prediction.
      </p>

      <div
        className="file-drop"
        onClick={() => ref.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={onDrop}
      >
        <input
          ref={ref} type="file" accept="image/*" style={{ display: 'none' }}
          onChange={e => e.target.files[0] && handleFile(e.target.files[0])}
        />
        <span className="drop-icon">📁</span>
        <p>{file ? file.name : 'Click or drag & drop an image'}</p>
        <small>PNG, JPG, GIF — resized to 32×32 RGB for the CNN</small>
      </div>

      {preview && (
        <img src={preview} alt="preview" className="preview-img" />
      )}

      <button className="btn-primary" onClick={handleSubmit} disabled={!file || loading}>
        {loading ? 'Analyzing…' : 'Predict & Explain'}
      </button>

      <div className="info-note">
        <strong>Tip:</strong> Use a clear photo of an airplane, car, animal, or other CIFAR-10 object.
        The model was trained on 50,000 color images across 10 object classes.
      </div>
    </div>
  );
}
