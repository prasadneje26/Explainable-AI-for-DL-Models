"""
app.py — FastAPI Backend: XAI for 4 Deep Learning Models
=========================================================
4 Models, 4 Input Types, SHAP Explanations

Endpoints:
  GET  /health
  GET  /models
  POST /predict/image    — Image CNN (CIFAR-10)
  POST /predict/text     — Text LSTM (IMDB Sentiment)
  POST /predict/tabular  — Tabular DNN (Iris)
  POST /predict/audio    — Audio 1D-CNN (Signal Classification)
"""

from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io, base64, json
from functools import lru_cache

import tensorflow as tf
from tensorflow.keras.datasets import imdb as keras_imdb

from models import load_model_for, MODEL_INFO
from data import (
    CIFAR10_CLASSES, SENTIMENT_LABELS, IRIS_CLASSES, AUDIO_CLASSES,
    preprocess_image, preprocess_text, preprocess_tabular, preprocess_audio,
    get_image_background, get_text_background,
    get_tabular_background, get_audio_background
)
from explainers import ImageSHAP, ImageLIME, ImageGradCAM, TextSHAP, TextLIME, TabularSHAP, AudioSHAP

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="XAI — 4 Models, 4 Input Types")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy model + explainer cache ──────────────────────────────────────────────

@lru_cache(maxsize=4)
def _model(t):
    return load_model_for(t)

@lru_cache(maxsize=4)
def _explainers(t):
    m = _model(t)
    if t == 'image':
        return {
            'shap': ImageSHAP(m, get_image_background(50)),
            'lime': ImageLIME(m),
            'gradcam': ImageGradCAM(m)
        }
    elif t == 'text':
        return {
            'shap': TextSHAP(m, get_text_background(50)),
            'lime': TextLIME(m)
        }
    elif t == 'tabular':
        return {'shap': TabularSHAP(m, get_tabular_background(50))}
    elif t == 'audio':
        return {'shap': AudioSHAP(m, get_audio_background(50))}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _top(probs, labels):
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), idx

def _all_preds(probs, labels):
    return [{"class": labels[i], "probability": float(probs[i])} for i in range(len(labels))]


# ── Routes ────────────────────────────────────────────────────────────────────

# It's perfectly safe to keep fast utility routes `async` since they do absolutely zero CPU blocking.
@app.get("/health")
async def health():
    return {"status": "ok", "models": list(MODEL_INFO.keys())}


@app.get("/models")
async def get_models():
    return {"models": MODEL_INFO}


# ── 1. Image ──────────────────────────────────────────────────────────────────

# Removed 'async' so FastAPI correctly pushes this highly CPU-bound task to worker pools
@app.post("/predict/image")
def predict_image(file: UploadFile = File(...)):
    """
    Input: image file (any format)
    Model: Image CNN (CIFAR-10)
    Output: predicted class + SHAP pixel heatmap
    """
    try:
        # Standard safety check: Limit image size to 5MB memory chunks
        raw = file.file.read()
        if len(raw) > 5 * 1024 * 1024:
            return {"success": False, "error": "Image file too large (Limit is 5MB)"}

        pil = Image.open(io.BytesIO(raw)).convert('RGB')
        inp = preprocess_image(np.array(pil))

        model = _model('image')
        probs = model.predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, CIFAR10_CLASSES)

        exps = _explainers('image')
        shap_overlay, shap_plot = exps['shap'].explain(inp, idx)
        lime_overlay = exps['lime'].explain(inp, idx)
        gradcam_heatmap = exps['gradcam'].explain(inp, idx)

        return {
            "success": True, "model": "Image CNN",
            "input_type": "Image", "dataset": "CIFAR-10",
            "prediction": label, "confidence": conf,
            "all_predictions": _all_preds(probs, CIFAR10_CLASSES),
            "explanations": {
                "shap": {
                    "overlay": shap_overlay,
                    "plot": shap_plot,
                    "description": "SHAP DeepExplainer highlighted pixels that most influenced predicting '{label}'."
                },
                "lime": {
                    "overlay": lime_overlay,
                    "description": "LIME highlighted superpixels that influenced the prediction."
                },
                "gradcam": {
                    "heatmap": gradcam_heatmap,
                    "description": "Grad-CAM shows the regions the CNN focused on for the prediction."
                }
            },
            "explanation": f"Multiple XAI methods explain why the model predicted '{label}'.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 2. Text ───────────────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str


@app.post("/predict/text")
def predict_text(body: TextInput):
    """
    Input: raw text string (movie review)
    Model: Text LSTM (IMDB Sentiment)
    Output: Positive/Negative + SHAP word importance chart
    """
    try:
        text = body.text.strip()
        if len(text) > 10000:
             return {"success": False, "error": "Text segment too long (Limit is 10,000 characters)"}

        inp  = preprocess_text(text)

        model = _model('text')
        probs = model.predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, SENTIMENT_LABELS)

        word_index = keras_imdb.get_word_index()
        exps = _explainers('text')
        shap_plot = exps['shap'].explain(inp, idx, text, word_index)
        lime_exp = exps['lime'].explain(text, idx)

        return {
            "success": True, "model": "Text LSTM",
            "input_type": "Text", "dataset": "IMDB Reviews",
            "prediction": label, "confidence": conf,
            "all_predictions": _all_preds(probs, SENTIMENT_LABELS),
            "explanations": {
                "shap": {
                    "plot": shap_plot,
                    "description": f"SHAP GradientExplainer shows which words most influenced the '{label}' prediction."
                },
                "lime": {
                    "explanation": lime_exp,
                    "description": "LIME shows the most important words for the prediction."
                }
            },
            "explanation": f"Multiple XAI methods explain why the model predicted '{label}'.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 3. Tabular ────────────────────────────────────────────────────────────────

class TabularInput(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float


@app.post("/predict/tabular")
def predict_tabular(body: TabularInput):
    """
    Input: 4 Iris measurements (cm)
    Model: Tabular DNN (Iris)
    Output: flower species + SHAP feature importance chart
    """
    try:
        features = [body.sepal_length, body.sepal_width,
                    body.petal_length, body.petal_width]
        inp = preprocess_tabular(features)

        model = _model('tabular')
        probs = model.predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, IRIS_CLASSES)

        exp = _explainer('tabular')
        plot_b64 = exp.explain(inp, idx, label)

        return {
            "success": True, "model": "Tabular DNN",
            "input_type": "Tabular / Structured Data", "dataset": "Iris Dataset",
            "prediction": label, "confidence": conf,
            "all_predictions": _all_preds(probs, IRIS_CLASSES),
            "shap_plot": plot_b64,
            "explanation": f"SHAP KernelExplainer shows which measurements most influenced predicting '{label}'.",
            "shap_description": "Green bars = features that pushed toward this species. Red bars = features that pushed against it."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 4. Audio ──────────────────────────────────────────────────────────────────

@app.post("/predict/audio")
def predict_audio(
    file: UploadFile = File(None),
    signal_type: str = Form(default='sine')
):
    """
    Input: .npy signal file OR generate a test signal (sine/square/noise)
    Model: Audio 1D-CNN
    Output: signal type + SHAP time-step importance plot
    """
    try:
        from data import AUDIO_LEN

        if file and file.filename:
            raw = file.file.read()
            if len(raw) > 5 * 1024 * 1024:
                return {"success": False, "error": "Signal file too large (Limit is 5MB)"}
            signal = np.load(io.BytesIO(raw)).flatten().astype(np.float32)
        else:
            # Generate a test signal
            t = np.linspace(0, 1, AUDIO_LEN)
            if signal_type == 'sine':
                signal = np.sin(2 * np.pi * 5 * t).astype(np.float32)
            elif signal_type == 'square':
                signal = np.sign(np.sin(2 * np.pi * 5 * t)).astype(np.float32)
            else:
                signal = np.random.normal(0, 1, AUDIO_LEN).astype(np.float32)

        inp = preprocess_audio(signal)

        model = _model('audio')
        probs = model.predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, AUDIO_CLASSES)

        exp = _explainer('audio')
        plot_b64 = exp.explain(inp, idx, label)

        return {
            "success": True, "model": "Audio 1D-CNN",
            "input_type": "Audio / Signal", "dataset": "Synthetic Signals",
            "prediction": label, "confidence": conf,
            "all_predictions": _all_preds(probs, AUDIO_CLASSES),
            "shap_plot": plot_b64,
            "explanation": f"SHAP GradientExplainer shows which time steps most influenced predicting '{label}'.",
            "shap_description": "Green areas = time steps that pushed toward this class. Red areas = pushed against it."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("Starting XAI Backend — 4 Models, 4 Input Types")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
