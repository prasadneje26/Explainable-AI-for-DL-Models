"""
data.py — Data loading and preprocessing for all 4 models.

Actual saved-model formats (inspected from .keras files):
  Image  : CIFAR-10 CNN — input (1, 32, 32, 3) float32 [0,1]
  Text   : Bidirectional LSTM — input (1, 200) int32 word-IDs
  Tabular: Iris DNN — input (1, 4) float32 standardized
  Audio  : 1D-CNN — input (1, 1000, 1) float32 normalized
"""

import re
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import joblib, os

# ── Labels ────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES  = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
SENTIMENT_LABELS = ['Negative', 'Positive']
IRIS_CLASSES     = ['setosa', 'versicolor', 'virginica']
AUDIO_CLASSES    = ['Sine Wave', 'Square Wave', 'Noise']

# Keep old name for backwards compatibility
MNIST_CLASSES = CIFAR10_CLASSES

VOCAB_SIZE    = 10000
SEQUENCE_LEN  = 200    # LSTM sequence length (matches saved model input shape)
AUDIO_LEN     = 1000   # Matches saved audio model input shape
SCALER_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_scaler.pkl')

# ── Cached globals ─────────────────────────────────────────────────────────────

_SCALER     = None
_WORD_INDEX = None


def _get_scaler():
    global _SCALER
    if _SCALER is None and os.path.exists(SCALER_PATH):
        _SCALER = joblib.load(SCALER_PATH)
    return _SCALER


def _get_word_index():
    global _WORD_INDEX
    if _WORD_INDEX is None:
        _WORD_INDEX = tf.keras.datasets.imdb.get_word_index()
    return _WORD_INDEX


# ── Model 1: CIFAR-10 Image ────────────────────────────────────────────────────

def load_cifar10():
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
    x_tr = x_tr.astype(np.float32) / 255.0         # (50000, 32, 32, 3)
    x_te = x_te.astype(np.float32) / 255.0
    y_tr = tf.keras.utils.to_categorical(y_tr, 10)
    y_te = tf.keras.utils.to_categorical(y_te, 10)
    return (x_tr, y_tr), (x_te, y_te)


def preprocess_image(image_input) -> np.ndarray:
    """
    Convert uploaded image → (1, 32, 32, 3) float32 [0,1].
    Converts to RGB and resizes to 32x32 to match CIFAR-10 CNN.
    """
    if isinstance(image_input, np.ndarray):
        pil = Image.fromarray(image_input.astype(np.uint8))
    else:
        pil = image_input
    pil = pil.convert('RGB').resize((32, 32), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr.reshape(1, 32, 32, 3)


def get_image_background(n=50) -> np.ndarray:
    (_, _), (x_te, _) = load_cifar10()
    idx = np.random.choice(len(x_te), n, replace=False)
    return x_te[idx]


# ── Model 2: Bidirectional LSTM (IMDB sequences) ────────────────────────────────

def text_to_sequence(text: str):
    """
    Convert raw text → (1, SEQUENCE_LEN) int32 padded word-ID sequence.
    Returns (seq_array, list_of_matched_words).
    IMDB word_index uses offset 3 (indices 0-3 reserved: padding, start, unknown).
    """
    word_index = _get_word_index()
    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    words = clean_text.split()

    seq = []
    matched_words = []
    for word in words:
        raw_idx = word_index.get(word)
        if raw_idx is not None:
            imdb_idx = raw_idx + 3        # IMDB offset
            if imdb_idx < VOCAB_SIZE:
                seq.append(imdb_idx)
                matched_words.append(word)

    # Left-pad or truncate to SEQUENCE_LEN
    seq = seq[-SEQUENCE_LEN:]
    seq = [0] * (SEQUENCE_LEN - len(seq)) + seq
    return np.array([seq], dtype=np.int32), matched_words


def preprocess_text(text: str) -> np.ndarray:
    seq, _ = text_to_sequence(text)
    return seq


def get_text_background(n=50) -> np.ndarray:
    """Return padded integer sequences from IMDB test set."""
    (_, _), (x_te, _) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    # Pad to SEQUENCE_LEN
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    x_padded = pad_sequences(x_te, maxlen=SEQUENCE_LEN, padding='pre', truncating='pre')
    idx = np.random.choice(len(x_padded), min(n, len(x_padded)), replace=False)
    return x_padded[idx].astype(np.int32)


# ── Model 3: Iris Tabular ──────────────────────────────────────────────────────

def load_iris_data():
    iris = load_iris()
    x = iris.data.astype(np.float32)
    y = tf.keras.utils.to_categorical(iris.target, 3)
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    global _SCALER
    _SCALER = scaler
    n   = len(x)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    return (x[idx[:split]], y[idx[:split]]), (x[idx[split:]], y[idx[split:]])


def preprocess_tabular(features: list) -> np.ndarray:
    arr    = np.array(features, dtype=np.float32).reshape(1, -1)
    scaler = _get_scaler()
    if scaler is not None:
        arr = scaler.transform(arr).astype(np.float32)
    return arr


def get_tabular_background(n=50) -> np.ndarray:
    iris = load_iris()
    x    = iris.data.astype(np.float32)
    scaler = _get_scaler()
    if scaler is not None:
        x = scaler.transform(x).astype(np.float32)
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return x[idx]


# ── Model 4: Synthetic Audio Signals ──────────────────────────────────────────

def generate_audio_dataset(n_per_class=800, length=AUDIO_LEN, seed=42):
    np.random.seed(seed)
    x, y = [], []
    t = np.linspace(0, 2 * np.pi, length)

    for _ in range(n_per_class):
        freq = np.random.uniform(3, 15)
        sig  = np.sin(freq * t) + np.random.normal(0, 0.02, length)
        x.append(sig.astype(np.float32)); y.append(0)

    for _ in range(n_per_class):
        freq = np.random.uniform(3, 15)
        sig  = np.sign(np.sin(freq * t)) + np.random.normal(0, 0.02, length)
        x.append(sig.astype(np.float32)); y.append(1)

    for _ in range(n_per_class):
        sig = np.random.normal(0, 1, length).astype(np.float32)
        x.append(sig); y.append(2)

    x   = np.array(x, dtype=np.float32)
    mx  = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
    x   = (x / mx)[..., np.newaxis]               # (N, AUDIO_LEN, 1)
    y_c = tf.keras.utils.to_categorical(y, 3)

    idx   = np.random.permutation(len(x))
    split = int(len(x) * 0.8)
    return (x[idx[:split]], y_c[idx[:split]]), (x[idx[split:]], y_c[idx[split:]])


def preprocess_audio(signal: np.ndarray) -> np.ndarray:
    """Resample to AUDIO_LEN, normalize, return (1, AUDIO_LEN, 1)."""
    if len(signal) != AUDIO_LEN:
        xo     = np.linspace(0, 1, len(signal))
        xn     = np.linspace(0, 1, AUDIO_LEN)
        signal = np.interp(xn, xo, signal)
    signal = signal.astype(np.float32)
    signal /= (np.max(np.abs(signal)) + 1e-8)
    return signal.reshape(1, AUDIO_LEN, 1)


def get_audio_background(n=50) -> np.ndarray:
    (_, _), (x_te, _) = generate_audio_dataset()
    idx = np.random.choice(len(x_te), min(n, len(x_te)), replace=False)
    return x_te[idx]
