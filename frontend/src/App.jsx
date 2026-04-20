import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ImageModel from './components/ImageModel';
import TextModel from './components/TextModel';
import TabularModel from './components/TabularModel';
import AudioModel from './components/AudioModel';
import ResultPanel from './components/ResultPanel';
import './App.css';

const API = '';

const TABS = [
  { id: 'image',   label: 'Image CNN',    icon: '🖼️',  desc: 'CIFAR-10 Recognition' },
  { id: 'text',    label: 'Text LSTM',    icon: '📝',  desc: 'Sentiment Analysis' },
  { id: 'tabular', label: 'Tabular DNN',  icon: '📊',  desc: 'Iris Classification' },
  { id: 'audio',   label: 'Audio 1D-CNN', icon: '🎵',  desc: 'Signal Classification' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  const handleResult = (res) => { setResult(res); setError(null); };
  const handleError  = (e)   => { setError(e);    setResult(null); };
  const handleLoad   = (v)   => { setLoading(v);  if (v) { setResult(null); setError(null); } };

  const switchTab = (id) => {
    setActiveTab(id);
    setResult(null);
    setError(null);
  };

  const inputProps = { api: API, onResult: handleResult, onError: handleError, onLoading: handleLoad, loading };

  return (
    <div className="App">
      <Navbar />

      <main className="main-content">
        <div className="container">

          <div className="tab-bar">
            {TABS.map(tab => (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => switchTab(tab.id)}
              >
                <span className="tab-icon">{tab.icon}</span>
                <span className="tab-label">{tab.label}</span>
                <span className="tab-desc">{tab.desc}</span>
              </button>
            ))}
          </div>

          <div className="content-grid">
            <div className="section input-section">
              {activeTab === 'image'   && <ImageModel   {...inputProps} />}
              {activeTab === 'text'    && <TextModel    {...inputProps} />}
              {activeTab === 'tabular' && <TabularModel {...inputProps} />}
              {activeTab === 'audio'   && <AudioModel   {...inputProps} />}
            </div>

            <div className="section results-section">
              {error   && <div className="error-banner">⚠ {error}</div>}

              {loading && (
                <div className="loading">
                  <div className="spinner" />
                  <p>Running model + computing SHAP explanations…</p>
                  <p className="loading-sub">This can take 15–60 s on CPU (KernelExplainer is slow)</p>
                </div>
              )}

              {result && !loading && <ResultPanel result={result} />}

              {!result && !loading && !error && (
                <div className="placeholder">
                  <span className="placeholder-icon">🔬</span>
                  <p>Fill in the input on the left and click <strong>Predict & Explain</strong></p>
                  <p style={{fontSize:'0.8rem', marginTop:'0.5rem', color:'#cbd5e1'}}>
                    Results will include the model prediction, class probabilities,<br/>SHAP visualizations, and a detailed explanation.
                  </p>
                </div>
              )}
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
