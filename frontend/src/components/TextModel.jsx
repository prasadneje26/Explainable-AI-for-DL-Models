import React, { useState } from 'react';
import axios from 'axios';

const EXAMPLES = [
  "This movie was absolutely fantastic! The acting was superb and the story was gripping.",
  "Terrible film. Boring plot, bad acting, complete waste of time.",
  "An average movie with some good moments but overall disappointing.",
];

export default function TextModel({ api, onResult, onError, onLoading, loading }) {
  const [text, setText] = useState('');

  const handleSubmit = async () => {
    if (!text.trim()) return;
    onLoading(true);
    try {
      const res = await axios.post(`${api}/predict/text`, { text });
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
      <h2>📝 Text LSTM</h2>
      <p style={{color:'#64748b',fontSize:'0.875rem',marginTop:0}}>
        Enter a movie review — classified as Positive or Negative sentiment
      </p>

      <div className="form-group">
        <label>Your Text</label>
        <textarea value={text} onChange={e => setText(e.target.value)}
          placeholder="Type a movie review here..." rows={5} />
      </div>

      <div style={{marginBottom:'1rem'}}>
        <label style={{fontSize:'0.8rem',color:'#64748b',fontWeight:600}}>Try an example:</label>
        <div style={{display:'flex',flexDirection:'column',gap:'0.4rem',marginTop:'0.4rem'}}>
          {EXAMPLES.map((ex, i) => (
            <button key={i} onClick={() => setText(ex)}
              style={{textAlign:'left',padding:'0.4rem 0.6rem',fontSize:'0.78rem',
                background:'#f1f5f9',border:'1px solid #e2e8f0',borderRadius:'6px',cursor:'pointer'}}>
              {ex.slice(0, 60)}...
            </button>
          ))}
        </div>
      </div>

      <button className="btn-primary" onClick={handleSubmit}
        disabled={!text.trim() || loading}>
        {loading ? 'Analyzing...' : 'Predict & Explain'}
      </button>
    </div>
  );
}
