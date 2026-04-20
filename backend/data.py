"""
data.py — Data loading and preprocessing for all 4 models.

Model 1 (Image):   MNIST — 28x28 grayscale digits
Model 2 (Text):    IMDB  — bag-of-words multi-hot vectors
Model 3 (Tabular): Iris  — 4 numerical features
Model 4 (Audio):   Synthetic signals — sine, square, noise
"""

import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import joblib, os

# ── Labels ────────────────────────────────────────────────────────────────────

MNIST_CLASSES    = [str(i) for i in range(10)]
SENTIMENT_LABELS = ['Negative', 'Positive']
IRIS_CLASSES     = ['setosa', 'versicolor', 'virginica']
AUDIO_CLASSES    = ['Sine Wave', 'Square Wave', 'Noise']

VOCAB_SIZE  = 10000
AUDIO_LEN   = 500
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iris_scaler.pkl')


# ── Model 1: MNIST ────────────────────────────────────────────────────────────

def load_mnist():
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    x_tr = x_tr.astype(np.float32)[..., np.newaxis] / 255.0   # (60000,28,28,1)
    x_te = x_te.astype(np.float32)[..., np.newaxis] / 255.0
    y_tr = tf.keras.utils.to_categorical(y_tr, 10)
    y_te = tf.keras.utils.to_categorical(y_te, 10)
    return (x_tr, y_tr), (x_te, y_te)


def preprocess_image(image_input) -> np.ndarray:
    """
    Convert uploaded image → (1, 28, 28, 1) float32 [0,1].
    Converts to grayscale and resizes to 28x28.
    """
    if isinstance(image_input, np.ndarray):
        pil = Image.fromarray(image_input.astype(np.uint8))
    else:
        pil = image_input
    pil = pil.convert('L').resize((28, 28))          # grayscale 28x28
    arr = np.array(pil, dtype=np.float32) / 255.0
    return arr.reshape(1, 28, 28, 1)


def get_image_background(n=100) -> np.ndarray:
    (_, _), (x_te, _) = load_mnist()
    idx = np.random.choice(len(x_te), n, replace=False)
    return x_te[idx]


# ── Model 2: IMDB Bag-of-Words ────────────────────────────────────────────────

def load_imdb_bow():
    """Load IMDB as multi-hot bag-of-words vectors."""
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

    def to_bow(sequences, dim=VOCAB_SIZE):
        out = np.zeros((len(sequences), dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            for idx in seq:
                if idx < dim:
                    out[i, idx] = 1.0
        return out

    x_tr = to_bow(x_tr)
    x_te = to_bow(x_te)
    y_tr = tf.keras.utils.to_categorical(y_tr, 2)
    y_te = tf.keras.utils.to_categorical(y_te, 2)
    return (x_tr, y_tr), (x_te, y_te)


def preprocess_text(text: str) -> np.ndarray:
    """
    Convert raw text → (1, VOCAB_SIZE) multi-hot float32 vector.
    Each position = 1 if that word appears in the text.
    """
    word_index = tf.keras.datasets.imdb.get_word_index()
    vec = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
    for word in text.lower().split():
        idx = word_index.get(word)
        if idx is not None and idx < VOCAB_SIZE:
            vec[0, idx] = 1.0
    return vec


def get_text_background(n=100) -> np.ndarray:
    (_, _), (x_te, _) = load_imdb_bow()
    idx = np.random.choice(len(x_te), n, replace=False)
    return x_te[idx]


# ── Model 3: Iris ─────────────────────────────────────────────────────────────

def load_iris_data():
    iris = load_iris()
    x = iris.data.astype(np.float32)
    y = tf.keras.utils.to_categorical(iris.target, 3)
    scaler = StandardScaler()
    x = scaler.fit_transform(x).astype(np.float32)
    joblib.dump(scaler, SCALER_PATH)
    n = len(x)
    idx = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    return (x[idx[:split]], y[idx[:split]]), (x[idx[split:]], y[idx[split:]])


def preprocess_tabular(features: list) -> np.ndarray:
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    if os.path.exists(SCALER_PATH):
        arr = joblib.load(SCALER_PATH).transform(arr).astype(np.float32)
    return arr


def get_tabular_background(n=50) -> np.ndarray:
    iris = load_iris()
    x = iris.data.astype(np.float32)
    if os.path.exists(SCALER_PATH):
        x = joblib.load(SCALER_PATH).transform(x).astype(np.float32)
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return x[idx]


# ── Model 4: Synthetic Audio Signals ─────────────────────────────────────────

def generate_audio_dataset(n_per_class=800, length=AUDIO_LEN, seed=42):
    """
    3 clearly distinct signal classes:
      0 = Sine wave   — smooth periodic
      1 = Square wave — abrupt +1/-1 transitions
      2 = Noise       — random Gaussian
    """
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

    x = np.array(x, dtype=np.float32)
    mx = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
    x = (x / mx)[..., np.newaxis]                   # (N, AUDIO_LEN, 1)
    y_cat = tf.keras.utils.to_categorical(y, 3)

    idx = np.random.permutation(len(x))
    split = int(len(x) * 0.8)
    return (x[idx[:split]], y_cat[idx[:split]]), (x[idx[split:]], y_cat[idx[split:]])


def preprocess_audio(signal: np.ndarray) -> np.ndarray:
    """Resample to AUDIO_LEN, normalize, return (1, AUDIO_LEN, 1)."""
    if len(signal) != AUDIO_LEN:
        xo = np.linspace(0, 1, len(signal))
        xn = np.linspace(0, 1, AUDIO_LEN)
        signal = np.interp(xn, xo, signal)
    signal = signal.astype(np.float32)
    signal /= (np.max(np.abs(signal)) + 1e-8)
    return signal.reshape(1, AUDIO_LEN, 1)


def get_audio_background(n=50) -> np.ndarray:
    (_, _), (x_te, _) = generate_audio_dataset()
    idx = np.random.choice(len(x_te), min(n, len(x_te)), replace=False)
    return x_te[idx]
