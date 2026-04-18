import React from 'react';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">XAI — Explainable AI</div>
        <div className="navbar-info">
          <span className="info-badge">TensorFlow</span>
          <span className="info-badge">SHAP</span>
          <span className="info-badge">4 Models</span>
          <span className="info-badge">4 Input Types</span>
        </div>
      </div>
    </nav>
  );
}
