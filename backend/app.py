"""
app.py — FastAPI Backend: XAI for 4 Deep Learning Models
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf

from models import load_model_for, MODEL_INFO
from data import (
    MNIST_CLASSES, SENTIMENT_LABELS, IRIS_CLASSES, AUDIO_CLASSES,
    preprocess_image, preprocess_text, preprocess_tabular, preprocess_audio,
    get_image_background, get_text_background,
    get_tabular_background, get_audio_background, AUDIO_LEN
)
from explainers import ImageSHAP, TextSHAP, TabularSHAP, AudioSHAP

app = FastAPI(title="XAI — 4 Deep Learning Models")
app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

_models, _explainers = {}, {}

def _model(t):
    if t not in _models:
        _models[t] = load_model_for(t)
    return _models[t]

def _exp(t):
    if t not in _explainers:
        m  = _model(t)
        bg = {'image': get_image_background, 'text': get_text_background,
              'tabular': get_tabular_background, 'audio': get_audio_background}[t]()
        _explainers[t] = {'image': ImageSHAP, 'text': TextSHAP,
                          'tabular': TabularSHAP, 'audio': AudioSHAP}[t](m, bg)
    return _explainers[t]

def _top(probs, labels):
    i = int(np.argmax(probs))
    return labels[i], float(probs[i]), i

def _all(probs, labels):
    return [{"class": labels[i], "probability": float(probs[i])} for i in range(len(labels))]


@app.get("/health")
async def health():
    return {"status": "ok", "models": list(MODEL_INFO.keys())}

@app.get("/models")
async def get_models():
    return {"models": MODEL_INFO}


# ── 1. Image ──────────────────────────────────────────────────────────────────

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        pil   = Image.open(io.BytesIO(await file.read()))
        inp   = preprocess_image(pil)
        probs = _model('image').predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, MNIST_CLASSES)

        # returns (overlay_b64, plot_b64, deep_text)
        overlay, plot, deep = _exp('image').explain(inp, idx)

        return {
            "success": True,
            "model": "Image CNN", "dataset": "MNIST",
            "input_type": "Image (Grayscale 28x28)",
            "shap_method": "DeepExplainer",
            "prediction": label, "confidence": conf,
            "all_predictions": _all(probs, MNIST_CLASSES),
            "shap_overlay": overlay,
            "shap_plot": plot,
            "explanation": f"SHAP DeepExplainer shows which pixels drove the prediction of digit '{label}'.",
            "shap_description": "Bright/hot pixels = most important for this digit. Dark = less relevant.",
            "deep_explanation": deep
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 2. Text ───────────────────────────────────────────────────────────────────

class TextInput(BaseModel):
    text: str

@app.post("/predict/text")
async def predict_text(body: TextInput):
    try:
        inp   = preprocess_text(body.text)
        probs = _model('text').predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, SENTIMENT_LABELS)

        # returns (plot_b64, deep_text)
        plot, deep = _exp('text').explain(inp, idx, body.text)

        return {
            "success": True,
            "model": "Text DNN", "dataset": "IMDB Reviews",
            "input_type": "Text (Bag-of-Words)",
            "shap_method": "KernelExplainer",
            "prediction": label, "confidence": conf,
            "all_predictions": _all(probs, SENTIMENT_LABELS),
            "shap_plot": plot,
            "explanation": f"SHAP KernelExplainer shows which words most influenced the '{label}' prediction.",
            "shap_description": "Green bars = words that pushed toward this sentiment. Red = pushed against.",
            "deep_explanation": deep
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
async def predict_tabular(body: TabularInput):
    try:
        raw   = [body.sepal_length, body.sepal_width,
                 body.petal_length, body.petal_width]
        inp   = preprocess_tabular(raw)
        probs = _model('tabular').predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, IRIS_CLASSES)

        # returns (plot_b64, deep_text) — raw needed for dual-axis chart
        plot, deep = _exp('tabular').explain(inp, idx, label, raw)

        return {
            "success": True,
            "model": "Tabular DNN", "dataset": "Iris Dataset",
            "input_type": "Tabular / Structured Data",
            "shap_method": "KernelExplainer",
            "prediction": label, "confidence": conf,
            "all_predictions": _all(probs, IRIS_CLASSES),
            "shap_plot": plot,
            "explanation": f"SHAP KernelExplainer shows which measurements drove the '{label}' prediction.",
            "shap_description": "Green bars = measurements that pushed toward this species. Red = pushed against.",
            "deep_explanation": deep
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── 4. Audio ──────────────────────────────────────────────────────────────────

@app.post("/predict/audio")
async def predict_audio(
    file: UploadFile = File(None),
    signal_type: str = Form(default='sine')
):
    try:
        if file and file.filename:
            signal = np.load(io.BytesIO(await file.read())).flatten().astype(np.float32)
        else:
            t = np.linspace(0, 2 * np.pi, AUDIO_LEN)
            signal = {
                'sine':   np.sin(5 * t),
                'square': np.sign(np.sin(5 * t)),
                'noise':  np.random.normal(0, 1, AUDIO_LEN)
            }.get(signal_type, np.sin(5 * t)).astype(np.float32)

        inp   = preprocess_audio(signal)
        probs = _model('audio').predict(inp, verbose=0)[0]
        label, conf, idx = _top(probs, AUDIO_CLASSES)

        # returns (plot_b64, deep_text)
        plot, deep = _exp('audio').explain(inp, idx, label)

        return {
            "success": True,
            "model": "Audio 1D-CNN", "dataset": "Synthetic Signals",
            "input_type": "Audio / Signal (1D Waveform)",
            "shap_method": "GradientExplainer",
            "prediction": label, "confidence": conf,
            "all_predictions": _all(probs, AUDIO_CLASSES),
            "shap_plot": plot,
            "explanation": f"SHAP GradientExplainer shows which time steps drove the '{label}' prediction.",
            "shap_description": "Green regions = time steps that pushed toward this class. Red = pushed against.",
            "deep_explanation": deep
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("Starting XAI Backend — 4 Deep Learning Models")
    print("Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
