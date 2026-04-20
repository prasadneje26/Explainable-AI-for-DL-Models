import React, { useState } from 'react';
import axios from 'axios';

const EXAMPLES = [
  { label: 'Positive review', text: "This movie was absolutely fantastic! The acting was superb and the story was gripping from start to finish. A masterpiece of cinema." },
  { label: 'Negative review', text: "Terrible film. Boring plot, bad acting and a complete waste of two hours. I would not recommend this to anyone." },
  { label: 'Mixed review',    text: "The visuals were stunning but the script was disappointing. Great soundtrack though. Average overall." },
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
      <p className="subtitle">
        Enter a movie review. The Bidirectional LSTM reads the text as a word sequence
        (up to 200 words) and classifies it as Positive or Negative.
        SHAP GradientExplainer reveals which words drove the sentiment decision.
      </p>

      <div className="form-group">
        <label>Review Text</label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Type or paste a movie review here…"
          rows={5}
        />
      </div>

      <div className="examples-section">
        <span className="examples-label">Load an example</span>
        <div className="example-btns">
          {EXAMPLES.map((ex, i) => (
            <button key={i} className="example-btn" onClick={() => setText(ex.text)}>
              <strong>{ex.label}:</strong>{' '}{ex.text.slice(0, 70)}…
            </button>
          ))}
        </div>
      </div>

      <button className="btn-primary" onClick={handleSubmit} disabled={!text.trim() || loading}>
        {loading ? 'Analyzing…' : 'Predict & Explain'}
      </button>

      <div className="info-note">
        <strong>Note:</strong> The LSTM reads up to 200 words as a sequence (word order matters!).
        Words not in the 10,000-word IMDB vocabulary are skipped.
        SHAP analysis takes ~30–60 s (GradientExplainer over embedding space).
      </div>
    </div>
  );
}
