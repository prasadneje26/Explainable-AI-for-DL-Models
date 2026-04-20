# XAI Course Project — Explainable AI for 4 Deep Learning Models

## Overview
A full-stack web application demonstrating how deep learning models make predictions across different data types (Image, Text, Tabular, Audio) with visual explanations via SHAP (Shapley Additive Explanations).

## Architecture

### Backend (FastAPI + Python)
- **Location**: `backend/`
- **Port**: 8000 (localhost only)
- **Entry point**: `backend/app.py`
- **Key files**:
  - `app.py` — FastAPI app with 4 prediction endpoints
  - `models.py` — Neural network architectures (CNN, DNN)
  - `explainers.py` — SHAP explanation logic
  - `data.py` — Data preprocessing utilities
  - `train.py` — Model training script
  - `*.keras` — Pre-trained model weights
  - `iris_scaler.pkl` — StandardScaler for Iris dataset

### Frontend (React)
- **Location**: `frontend/`
- **Port**: 5000 (0.0.0.0)
- **Framework**: Create React App (react-scripts)
- **Proxy**: Forwards `/predict/*` and `/models` to backend at `http://localhost:8000`

## Models
| Data Type | Model | Dataset | Task |
|-----------|-------|---------|------|
| Image | 2D-CNN | MNIST | Handwritten Digit Recognition |
| Text | DNN (Bag-of-Words) | IMDB | Sentiment Analysis |
| Tabular | DNN | Iris | Flower Species Classification |
| Audio | 1D-CNN | Synthetic | Signal Type Classification |

## Workflows
- **Backend API**: `cd backend && python -m uvicorn app:app --host 127.0.0.1 --port 8000`
- **Start application**: `cd frontend && HOST=0.0.0.0 PORT=5000 BROWSER=none npm start`

## Dependencies
- **Python**: fastapi, uvicorn, tensorflow, shap, scikit-learn, numpy, pillow, matplotlib, opencv-python, lime
- **Node.js**: react, react-dom, react-scripts, axios

## Key Notes
- CORS is configured to allow all origins (`*`) for development
- Frontend uses empty string for API base URL (relies on CRA proxy)
- Backend TF models load lazily on first request (may take 10-30s for first inference)
