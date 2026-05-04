"""
models.py — 4 Deep Learning Models for 4 Input Types

Actual saved model formats (do NOT change without retraining):
  Image  : CIFAR-10 CNN   — input (32,32,3) RGB, 10 classes
  Text   : Bidirectional LSTM — input (200,) int32 word-IDs, 2 classes
  Tabular: Iris DNN       — input (4,) standardized, 3 classes
  Audio  : 1D-CNN         — input (1000,1) waveform, 3 classes
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models

_DIR = os.path.dirname(os.path.abspath(__file__))

NUM_CIFAR10 = 10
NUM_IMDB    = 2
NUM_IRIS    = 3
NUM_AUDIO   = 3
VOCAB_SIZE  = 10000
SEQUENCE_LEN= 200
AUDIO_LEN   = 1000


def build_image_cnn():
    """
    CNN for CIFAR-10 — 3 conv blocks with BatchNorm for stable training.
    Input : (32,32,3) RGB [0,1]
    Output: 10-class softmax
    Target: ~85-90% accuracy
    """
    return models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(32, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2), layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same'), layers.BatchNormalization(), layers.Activation('relu'),
        layers.MaxPooling2D(2), layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.4),
        layers.Dense(NUM_CIFAR10, activation='softmax')
    ], name='ImageCNN_CIFAR10')


def build_text_dnn():
    """
    BiLSTM for IMDB sentiment.
    Input : (SEQUENCE_LEN,) integer word-ID sequence
    Output: 2-class softmax (Negative / Positive)
    Target: ~87% accuracy
    """
    return models.Sequential([
        layers.Input(shape=(SEQUENCE_LEN,), dtype='int32'),
        layers.Embedding(VOCAB_SIZE, 64, input_length=SEQUENCE_LEN),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(NUM_IMDB, activation='softmax')
    ], name='TextLSTM_IMDB')


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
        'name': 'Image CNN', 'input_type': 'Image (RGB)',
        'dataset': 'CIFAR-10', 'task': 'Object Recognition (10 classes)',
        'architecture': 'Conv2D(64)×2 → Pool → Conv2D(128)×2 → Pool → Conv2D(256) → Dense(512) → Softmax',
        'input_desc': 'Upload any photo — airplane, car, animal, ship, truck, etc.',
        'theory': (
            'A Convolutional Neural Network (CNN) applies learnable 3×3 filters across a '
            '32×32 RGB image to detect spatial features. Early layers detect color edges and '
            'textures; deeper layers learn complex object parts (wings, wheels, fur). '
            'Three convolutional blocks progressively grow from 64→128→256 channels. '
            'BatchNormalization stabilizes gradient flow, Dropout prevents overfitting on '
            'CIFAR-10\'s 50,000 training images across 10 object categories.'
        )
    },
    'text': {
        'name': 'Text LSTM', 'input_type': 'Text',
        'dataset': 'IMDB Movie Reviews', 'task': 'Sentiment Analysis (Positive / Negative)',
        'architecture': 'Embedding(10000,64) → BiLSTM(128) → BiLSTM(64) → Dense(64) → Softmax',
        'input_desc': 'Type any movie review or short opinion text',
        'theory': (
            'A Bidirectional LSTM processes text as a sequence of up to 200 word IDs. '
            'The Embedding layer maps each word to a learned 64-dimensional vector in semantic '
            'space (similar words cluster together). The two BiLSTM layers read the sequence '
            'both left-to-right and right-to-left simultaneously, capturing long-range context '
            'that a simple word-count model would miss (e.g., "not bad" vs "not good"). '
            'Trained on 25,000 IMDB reviews, it achieves ~87% sentiment accuracy.'
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
