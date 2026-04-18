"""
data.py — Data loading and preprocessing for all 4 input types.

Model 1 (Image):   CIFAR-10 — 32x32 RGB images
Model 2 (Text):    IMDB     — movie review sequences
Model 3 (Tabular): Iris     — 4 numerical features
Model 4 (Audio):   Synthetic signals — sine, square, noise
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# ── Labels ────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES  = ('airplane','automobile','bird','cat','deer',
                    'dog','frog','horse','ship','truck')
SENTIMENT_LABELS = ('Negative', 'Positive')
IRIS_CLASSES     = ('setosa', 'versicolor', 'virginica')
AUDIO_CLASSES    = ('Sine Wave', 'Square Wave', 'Noise')

VOCAB_SIZE  = 10000
MAX_SEQ_LEN = 200
AUDIO_LEN   = 1000

SCALER_PATH = 'iris_scaler.pkl'


# ── Model 1: Image ────────────────────────────────────────────────────────────

def load_cifar10_data():
    (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
    x_tr = x_tr.astype(np.float32) / 255.0
    x_te = x_te.astype(np.float32) / 255.0
    y_tr = tf.keras.utils.to_categorical(y_tr, 10)
    y_te = tf.keras.utils.to_categorical(y_te, 10)
    return (x_tr, y_tr), (x_te, y_te)


def preprocess_image(image_input) -> np.ndarray:
    """
    Preprocess an uploaded image for the Image CNN.
    Returns (1, 32, 32, 3) float32 array normalized to [0,1].
    """
    if isinstance(image_input, np.ndarray):
        pil = Image.fromarray(image_input.astype(np.uint8)).convert('RGB')
    else:
        pil = image_input.convert('RGB')
    # Use anti-aliasing (LANCZOS) to prevent high-res images from turning into jagged geometric shapes
    pil = pil.resize((32, 32), Image.Resampling.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def get_image_background(n=50) -> np.ndarray:
    """Background samples for SHAP (image model)."""
    (_, _), (x_te, _) = cifar10.load_data()
    idx = np.random.choice(len(x_te), n, replace=False)
    bg = x_te[idx].astype(np.float32) / 255.0
    return bg


# ── Model 2: Text ─────────────────────────────────────────────────────────────

def load_imdb_data():
    (x_tr, y_tr), (x_te, y_te) = imdb.load_data(num_words=VOCAB_SIZE)
    x_tr = pad_sequences(x_tr, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    x_te = pad_sequences(x_te, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    y_tr = tf.keras.utils.to_categorical(y_tr, 2)
    y_te = tf.keras.utils.to_categorical(y_te, 2)
    return (x_tr, y_tr), (x_te, y_te)


def preprocess_text(text: str) -> np.ndarray:
    """
    Convert raw text to padded integer sequence for the Text LSTM.
    Returns (1, MAX_SEQ_LEN) int32 array.
    """
    word_index = imdb.get_word_index()
    # IMDB word index is offset by 3 (reserved tokens)
    tokens = []
    for word in text.lower().split():
        idx = word_index.get(word, 2)  # 2 = unknown
        if idx < VOCAB_SIZE:
            tokens.append(idx + 3)
        else:
            tokens.append(2)
    seq = pad_sequences([tokens], maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    return seq.astype(np.int32)


def get_text_background(n=50) -> np.ndarray:
    """Background samples for SHAP (text model)."""
    (_, _), (x_te, _) = imdb.load_data(num_words=VOCAB_SIZE)
    idx = np.random.choice(len(x_te), n, replace=False)
    x_te_slice = pad_sequences(x_te[idx], maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    return x_te_slice.astype(np.float32)


# ── Model 3: Tabular ──────────────────────────────────────────────────────────

def load_iris_data():
    iris = load_iris()
    x = iris.data.astype(np.float32)
    y = tf.keras.utils.to_categorical(iris.target, 3)

    # Fit and save scaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)

    # 80/20 split
    n = len(x_scaled)
    split = int(n * 0.8)
    idx = np.random.permutation(n)
    return (x_scaled[idx[:split]], y[idx[:split]]), (x_scaled[idx[split:]], y[idx[split:]])


def preprocess_tabular(features: list) -> np.ndarray:
    """
    Preprocess 4 Iris features for the Tabular DNN.
    features: [sepal_length, sepal_width, petal_length, petal_width]
    Returns (1, 4) float32 array (StandardScaler normalized).
    """
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        arr = scaler.transform(arr).astype(np.float32)
    return arr


def get_tabular_background(n=50) -> np.ndarray:
    """Background samples for SHAP (tabular model)."""
    iris = load_iris()
    x = iris.data.astype(np.float32)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        x = scaler.transform(x).astype(np.float32)
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return x[idx]


# ── Model 4: Audio ────────────────────────────────────────────────────────────

def generate_audio_dataset(n_per_class=1000, length=AUDIO_LEN, seed=42):
    """
    Generate synthetic audio signals for 3 clearly distinct classes:
      0 = Sine wave  — smooth periodic signal
      1 = Square wave — abrupt transitions between +1 and -1
      2 = Noise       — random Gaussian signal (no periodicity)
    """
    np.random.seed(seed)
    x, y = [], []
    t = np.linspace(0, 2 * np.pi, length)

    for _ in range(n_per_class):
        freq = np.random.uniform(3, 15)
        # Pure sine — very small noise so class is distinguishable
        sig = np.sin(freq * t) + np.random.normal(0, 0.02, length)
        x.append(sig.astype(np.float32)); y.append(0)

    for _ in range(n_per_class):
        freq = np.random.uniform(3, 15)
        # Square wave — sign of sine, very distinct shape
        sig = np.sign(np.sin(freq * t)) + np.random.normal(0, 0.02, length)
        x.append(sig.astype(np.float32)); y.append(1)

    for _ in range(n_per_class):
        # Pure Gaussian noise — no structure
        sig = np.random.normal(0, 1, length).astype(np.float32)
        x.append(sig); y.append(2)

    x = np.array(x, dtype=np.float32)
    y_cat = tf.keras.utils.to_categorical(y, 3)

    # Normalize each signal to [-1, 1]
    mx = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
    x = x / mx
    x = x[..., np.newaxis]  # (N, AUDIO_LEN, 1)

    idx = np.random.permutation(len(x))
    split = int(len(x) * 0.8)
    return (x[idx[:split]], y_cat[idx[:split]]), (x[idx[split:]], y_cat[idx[split:]])


def preprocess_audio(signal: np.ndarray) -> np.ndarray:
    """
    Preprocess a 1D signal array for the Audio CNN.
    signal: 1D numpy array of any length → resampled to AUDIO_LEN
    Returns (1, AUDIO_LEN, 1) float32 array.
    """
    # Resample to AUDIO_LEN using linear interpolation
    if len(signal) != AUDIO_LEN:
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, AUDIO_LEN)
        signal = np.interp(x_new, x_old, signal)
    signal = signal.astype(np.float32)
    # Normalize
    mx = np.max(np.abs(signal)) + 1e-8
    signal = signal / mx
    return signal.reshape(1, AUDIO_LEN, 1)


def get_audio_background(n=50) -> np.ndarray:
    """Background samples for SHAP (audio model)."""
    (_, _), (x_te, _) = generate_audio_dataset()
    idx = np.random.choice(len(x_te), min(n, len(x_te)), replace=False)
    return x_te[idx]
