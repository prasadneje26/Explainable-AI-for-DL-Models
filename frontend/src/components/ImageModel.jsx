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
      <p style={{color:'#64748b',fontSize:'0.875rem',marginTop:0}}>
        Upload a handwritten digit image — recognized as 0 through 9
      </p>

      <div className="file-drop" onClick={() => ref.current?.click()}>
        <input ref={ref} type="file" accept="image/*" style={{display:'none'}}
          onChange={e => e.target.files[0] && handleFile(e.target.files[0])} />
        <p style={{fontSize:'2rem',margin:0}}>📁</p>
        <p>Click to upload image</p>
        <p style={{fontSize:'0.75rem'}}>PNG, JPG, GIF</p>
      </div>

      {preview && <img src={preview} alt="preview" className="preview-img" />}

      <button className="btn-primary" onClick={handleSubmit}
        disabled={!file || loading}>
        {loading ? 'Analyzing...' : 'Predict & Explain'}
      </button>

      <div style={{marginTop:'1rem',fontSize:'0.8rem',color:'#64748b'}}>
        <strong>Classes:</strong> Digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        <br/><strong>Tip:</strong> Draw a digit on paper, take a photo, or use any MNIST-style image
      </div>
    </div>
  );
}
