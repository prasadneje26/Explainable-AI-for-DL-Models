"""
models.py — 4 Deep Learning Models for 4 Different Input Types

Model 1: Image CNN        — CIFAR-10 image classification (32x32 RGB)
Model 2: Text LSTM        — IMDB sentiment analysis (text sequences)
Model 3: Tabular DNN      — Iris flower classification (structured data)
Model 4: Audio 1D-CNN     — Synthetic audio signal classification (1D waveform)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model

NUM_CLASSES_IMAGE   = 10   # CIFAR-10
NUM_CLASSES_TEXT    = 2    # Positive / Negative
NUM_CLASSES_TABULAR = 3    # Iris: setosa, versicolor, virginica
NUM_CLASSES_AUDIO   = 3    # 3 synthetic signal types

VOCAB_SIZE    = 10000
MAX_SEQ_LEN   = 200
EMBED_DIM     = 64
AUDIO_LEN     = 1000  # 1000 time steps per signal


# ── Model 1: Image CNN (CIFAR-10) ─────────────────────────────────────────────

def build_image_cnn(input_shape=(32, 32, 3), num_classes=NUM_CLASSES_IMAGE):
    """
    Custom CNN for image classification on CIFAR-10.
    Input: (32, 32, 3) RGB image normalized to [0,1]
    Output: 10-class softmax (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)
    """
    model = models.Sequential([
        layers.Conv2D(64, (3,3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.3),

        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.3),

        layers.Conv2D(256, (3,3), padding='same'),
        layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2,2), layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='ImageCNN')
    return model


# ── Model 2: Text LSTM (IMDB Sentiment) ──────────────────────────────────────

def build_text_lstm(vocab_size=VOCAB_SIZE, max_len=MAX_SEQ_LEN,
                    embed_dim=EMBED_DIM, num_classes=NUM_CLASSES_TEXT):
    """
    Bidirectional LSTM for text sentiment analysis on IMDB dataset.
    Input: integer sequence of word indices, length MAX_SEQ_LEN
    Output: 2-class softmax (Negative=0, Positive=1)

    Architecture:
      Embedding → BiLSTM → BiLSTM → Dense → Softmax
    """
    model = models.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=max_len),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3)),
        layers.Bidirectional(layers.LSTM(32, dropout=0.3)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name='TextLSTM')
    return model


# ── Model 3: Tabular DNN (Iris) ───────────────────────────────────────────────

def build_tabular_dnn(input_dim=4, num_classes=NUM_CLASSES_TABULAR):
    """
    Optimum Network for tabular/structured data classification on Iris dataset.
    Input: 4 numerical features (sepal length, sepal width, petal length, petal width)
    Output: 3-class softmax (setosa, versicolor, virginica)
    """
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='TabularDNN')
    return model


# ── Model 4: Audio 1D-CNN (Signal Classification) ────────────────────────────

def build_audio_cnn(input_len=AUDIO_LEN, num_classes=NUM_CLASSES_AUDIO):
    """
    1D CNN for audio/signal classification.
    Input: 1D waveform of length AUDIO_LEN (1000 time steps)
    Output: 3-class softmax (Sine Wave, Square Wave, Noise)
    """
    inp = layers.Input(shape=(input_len, 1))

    x = layers.Conv1D(64, 7, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(4)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out, name='AudioCNN')


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_BUILDERS = {
    'image':    build_image_cnn,
    'text':     build_text_lstm,
    'tabular':  build_tabular_dnn,
    'audio':    build_audio_cnn,
}

MODEL_SAVE_PATHS = {
    'image':    'saved_image.keras',
    'text':     'saved_text.keras',
    'tabular':  'saved_tabular.keras',
    'audio':    'saved_audio.keras',
}

MODEL_INFO = {
    'image': {
        'name': 'Image CNN',
        'input_type': 'Image',
        'dataset': 'CIFAR-10',
        'task': 'Image Classification',
        'input_desc': 'Upload any image (airplane, car, bird, cat, deer, dog, frog, horse, ship, truck)',
        'architecture': 'Custom CNN with BatchNorm + Dropout'
    },
    'text': {
        'name': 'Text LSTM',
        'input_type': 'Text',
        'dataset': 'IMDB Reviews',
        'task': 'Sentiment Analysis',
        'input_desc': 'Enter a movie review or any text to classify as Positive or Negative',
        'architecture': 'Bidirectional LSTM with Embedding layer'
    },
    'tabular': {
        'name': 'Tabular DNN',
        'input_type': 'Tabular / Structured',
        'dataset': 'Iris Dataset',
        'task': 'Flower Species Classification',
        'input_desc': 'Enter 4 measurements: sepal length, sepal width, petal length, petal width (in cm)',
        'architecture': 'Deep Neural Network with BatchNorm'
    },
    'audio': {
        'name': 'Audio 1D-CNN',
        'input_type': 'Audio / Signal',
        'dataset': 'Synthetic Signals',
        'task': 'Signal Type Classification',
        'input_desc': 'Upload a .npy signal file or use the test generator',
        'architecture': '1D CNN with GlobalAveragePooling'
    }
}


def get_model(model_type: str):
    t = model_type.lower()
    if t not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model type '{t}'. Choose from: {list(MODEL_BUILDERS.keys())}")
    return MODEL_BUILDERS[t]()


def save_model(model, model_type: str):
    path = MODEL_SAVE_PATHS[model_type.lower()]
    model.save(path)
    print(f"[OK] Saved: {path}")


def load_model_for(model_type: str):
    """Load trained model or return fresh model with random weights."""
    t = model_type.lower()
    path = MODEL_SAVE_PATHS.get(t)
    if path and os.path.exists(path):
        m = tf.keras.models.load_model(path)
        print(f"[OK] Loaded trained model: {path}")
    else:
        m = get_model(t)
        print(f"[WARN] No saved weights for '{t}' — using untrained model")
    return m
