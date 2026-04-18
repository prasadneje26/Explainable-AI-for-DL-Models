import React, { useState } from 'react';
import Navbar from './components/Navbar';
import ImageModel from './components/ImageModel';
import TextModel from './components/TextModel';
import TabularModel from './components/TabularModel';
import AudioModel from './components/AudioModel';
import ResultPanel from './components/ResultPanel';
import ExplanationView from './components/ExplanationView';
import './App.css';

const API = 'http://localhost:8000';

const TABS = [
  { id: 'image',   label: 'Image CNN',     icon: '🖼️',  desc: 'CIFAR-10 Classification' },
  { id: 'text',    label: 'Text LSTM',     icon: '📝',  desc: 'Sentiment Analysis' },
  { id: 'tabular', label: 'Tabular DNN',   icon: '📊',  desc: 'Iris Classification' },
  { id: 'audio',   label: 'Audio 1D-CNN',  icon: '🎵',  desc: 'Signal Classification' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);

  const handleResult = (res) => { setResult(res); setError(null); };
  const handleError  = (e)   => { setError(e);    setResult(null); };
  const handleLoad   = (v)   => { setLoading(v); if (v) { setResult(null); setError(null); } };

  const switchTab = (id) => {
    setActiveTab(id);
    setResult(null);
    setError(null);
  };

  return (
    <div className="App">
      <Navbar />

      <main className="main-content">
        <div className="container">


          {/* Tab bar */}
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
            {/* Left: Input panel */}
            <div className="section input-section">
              {activeTab === 'image'   && <ImageModel   api={API} onResult={handleResult} onError={handleError} onLoading={handleLoad} loading={loading} />}
              {activeTab === 'text'    && <TextModel    api={API} onResult={handleResult} onError={handleError} onLoading={handleLoad} loading={loading} />}
              {activeTab === 'tabular' && <TabularModel api={API} onResult={handleResult} onError={handleError} onLoading={handleLoad} loading={loading} />}
              {activeTab === 'audio'   && <AudioModel   api={API} onResult={handleResult} onError={handleError} onLoading={handleLoad} loading={loading} />}
            </div>

            {/* Right: Result panel */}
            <div className="section results-section">
              {error   && <div className="error-banner">Error: {error}</div>}
              {loading && (
                <div className="loading">
                  <div className="spinner"></div>
                  <p>Running model + generating explanations...</p>
                  <p className="loading-sub">May take 15–30s on CPU</p>
                </div>
              )}
              {result && !loading && <ResultPanel result={result} />}
              {result && !loading && result.explanations && <ExplanationView explanations={result.explanations} />}
              {!result && !loading && !error && (
                <div className="placeholder">
                  <p>Fill in the input on the left and click Predict</p>
                </div>
              )}
            </div>
          </div>


        </div>
      </main>

    </div>
  );
}
