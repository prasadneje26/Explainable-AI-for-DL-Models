"""
models.py — 4 Deep Learning Models for 4 Input Types

Model 1: Image CNN    — MNIST digit recognition (28x28 grayscale)
Model 2: Text DNN     — IMDB sentiment analysis (bag-of-words)
Model 3: Tabular DNN  — Iris flower classification (4 features)
Model 4: Audio 1D-CNN — Synthetic signal classification (sine/square/noise)
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Always save/load models relative to this file's directory (backend/)
_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_MNIST  = 10
NUM_IMDB   = 2
NUM_IRIS   = 3
NUM_AUDIO  = 3
VOCAB_SIZE = 10000
AUDIO_LEN  = 500


def build_image_cnn():
    """
    CNN for MNIST — 3 conv blocks with BatchNorm for stable training.
    Input : (28,28,1) grayscale [0,1]
    Output: 10-class softmax
    Target: ~99.5% accuracy
    """
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(NUM_MNIST, activation='softmax')
    ], name='ImageCNN_MNIST')


def build_text_dnn():
    """
    DNN for IMDB sentiment — wider layers + L2 regularization.
    Input : (VOCAB_SIZE,) multi-hot bag-of-words
    Output: 2-class softmax (Negative / Positive)
    Target: ~90% accuracy
    """
    reg = tf.keras.regularizers.l2(1e-4)
    return models.Sequential([
        layers.Input(shape=(VOCAB_SIZE,)),
        layers.Dense(256, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.4),
        layers.Dense(64,  activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.3),
        layers.Dense(NUM_IMDB, activation='softmax')
    ], name='TextDNN_IMDB')


def build_tabular_dnn():
    """
    DNN for Iris — deep enough to learn non-linear boundaries.
    Input : (4,) standardized features
    Output: 3-class softmax
    Target: ~100% accuracy
    """
    return models.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64,  activation='relu'), layers.BatchNormalization(), layers.Dropout(0.2),
        layers.Dense(32,  activation='relu'),
        layers.Dense(NUM_IRIS, activation='softmax')
    ], name='TabularDNN_Iris')


def build_audio_cnn():
    """
    1D-CNN for signal classification — 3 conv blocks + GAP.
    Input : (AUDIO_LEN, 1) normalized waveform
    Output: 3-class softmax (Sine / Square / Noise)
    Target: ~100% accuracy
    """
    return models.Sequential([
        layers.Input(shape=(AUDIO_LEN, 1)),
        layers.Conv1D(32, 7, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling1D(4), layers.Dropout(0.2),

        layers.Conv1D(64, 5, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling1D(4), layers.Dropout(0.2),

        layers.Conv1D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.GlobalAveragePooling1D(),

        layers.Dense(64, activation='relu'), layers.Dropout(0.3),
        layers.Dense(NUM_AUDIO, activation='softmax')
    ], name='AudioCNN_Signals')


MODEL_BUILDERS = {
    'image': build_image_cnn, 'text': build_text_dnn,
    'tabular': build_tabular_dnn, 'audio': build_audio_cnn,
}

MODEL_SAVE_PATHS = {
    'image':   os.path.join(_DIR, 'saved_image.keras'),
    'text':    os.path.join(_DIR, 'saved_text.keras'),
    'tabular': os.path.join(_DIR, 'saved_tabular.keras'),
    'audio':   os.path.join(_DIR, 'saved_audio.keras'),
}

MODEL_INFO = {
    'image': {
        'name': 'Image CNN', 'input_type': 'Image (Grayscale)',
        'dataset': 'MNIST', 'task': 'Handwritten Digit Recognition (0–9)',
        'architecture': 'Conv2D×2 → Pool → Conv2D×2 → Pool → Dense(256) → Softmax',
        'input_desc': 'Upload a handwritten digit image (0–9)',
        'theory': (
            'A Convolutional Neural Network (CNN) uses learnable filters that slide over '
            'the image to detect local patterns like edges, curves, and strokes. '
            'Early layers detect simple features (edges), deeper layers detect complex '
            'shapes (loops in 8, vertical stroke in 1). MaxPooling reduces spatial size '
            'while keeping important features. BatchNormalization stabilizes training.'
        )
    },
    'text': {
        'name': 'Text DNN', 'input_type': 'Text',
        'dataset': 'IMDB Movie Reviews', 'task': 'Sentiment Analysis (Positive / Negative)',
        'architecture': 'BoW Vector → Dense(256) → Dense(128) → Dense(64) → Softmax',
        'input_desc': 'Type any movie review or sentence',
        'theory': (
            'Bag-of-Words (BoW) converts text into a fixed-size vector where each position '
            'represents a word from the vocabulary. If the word appears in the text, that '
            'position is 1, otherwise 0. The Dense Neural Network then learns which word '
            'combinations indicate positive or negative sentiment. L2 regularization '
            'prevents overfitting on the 50,000 IMDB reviews.'
        )
    },
    'tabular': {
        'name': 'Tabular DNN', 'input_type': 'Tabular / Structured Data',
        'dataset': 'Iris Flower Dataset', 'task': 'Flower Species Classification',
        'architecture': 'Dense(128) → BN → Dense(64) → BN → Dense(32) → Softmax',
        'input_desc': 'Enter 4 flower measurements in cm',
        'theory': (
            'A Deep Neural Network for tabular data learns non-linear decision boundaries '
            'between classes. The 4 Iris features (sepal/petal length/width) are '
            'standardized using StandardScaler so all features contribute equally. '
            'BatchNormalization after each layer ensures stable gradient flow. '
            'The model learns that petal measurements are the most discriminative features.'
        )
    },
    'audio': {
        'name': 'Audio 1D-CNN', 'input_type': 'Audio / Signal (1D Waveform)',
        'dataset': 'Synthetic Signals', 'task': 'Signal Type Classification',
        'architecture': 'Conv1D(32) → Conv1D(64) → Conv1D(128) → GAP → Dense → Softmax',
        'input_desc': 'Generate a test signal or upload a .npy file',
        'theory': (
            'A 1D-CNN applies convolutional filters along the time axis of a signal. '
            'It learns to detect temporal patterns: smooth periodicity (sine), '
            'abrupt transitions (square), or random fluctuations (noise). '
            'GlobalAveragePooling aggregates features across all time steps into a '
            'single vector, making the model robust to signal length variations.'
        )
    }
}


def get_model(t: str):
    t = t.lower()
    if t not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{t}'")
    return MODEL_BUILDERS[t]()


def load_model_for(t: str):
    t = t.lower()
    path = MODEL_SAVE_PATHS.get(t)
    if path and os.path.exists(path):
        m = tf.keras.models.load_model(path)
        print(f"[OK] Loaded: {path}")
    else:
        m = get_model(t)
        print(f"[WARN] No saved weights for '{t}'")
    return m
