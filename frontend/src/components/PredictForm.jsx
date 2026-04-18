import React, { useRef } from 'react';
import './PredictForm.css';

function PredictForm({ onFileSelect, imagePreview, onPredict, onReset, loading }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) onFileSelect(file);
  };

  return (
    <div className="predict-form">
      <h2>Upload Image</h2>

      <div className="file-input-area" onClick={() => fileInputRef.current?.click()}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <div className="upload-icon">📁</div>
        <p className="upload-text">Click to upload image</p>
        <p className="upload-hint">PNG, JPG, GIF — any size</p>
      </div>

      {imagePreview && (
        <div className="image-preview">
          <img src={imagePreview} alt="preview" />
        </div>
      )}

      <div className="auto-info">
        <span className="auto-badge">Auto</span>
        Model is selected automatically based on your image
      </div>

      <div className="button-group">
        <button
          onClick={onPredict}
          disabled={!imagePreview || loading}
          className="btn btn-primary"
        >
          {loading ? 'Analyzing...' : 'Analyze & Explain'}
        </button>
        {imagePreview && (
          <button onClick={onReset} disabled={loading} className="btn btn-secondary">
            Reset
          </button>
        )}
      </div>

      <div className="instructions">
        <h4>Supported classes:</h4>
        <p>airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck</p>
      </div>
    </div>
  );
}

export default PredictForm;
