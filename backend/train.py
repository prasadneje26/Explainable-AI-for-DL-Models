"""
train.py — Train all 4 models on their respective datasets.

Usage:
    python train.py --model image     # Image CNN on CIFAR-10
    python train.py --model text      # Text LSTM on IMDB
    python train.py --model tabular   # Tabular DNN on Iris
    python train.py --model audio     # Audio 1D-CNN on synthetic signals
    python train.py --model all       # Train all 4 sequentially
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import get_model, save_model, MODEL_SAVE_PATHS
from data import (
    load_cifar10_data, load_imdb_data, load_iris_data,
    generate_audio_dataset
)


def get_callbacks(model_type: str):
    return [
        callbacks.ModelCheckpoint(
            MODEL_SAVE_PATHS[model_type],
            save_best_only=True, monitor='val_accuracy', verbose=1
        ),
        callbacks.EarlyStopping(
            patience=8, restore_best_weights=True,
            monitor='val_accuracy', verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=2,
            min_lr=1e-6, verbose=1
        )
    ]


# ── Model 1: Image CNN ────────────────────────────────────────────────────────

def train_image():
    print("\n=== Training Image CNN (CIFAR-10) ===")
    (x_tr, y_tr), (x_te, y_te) = load_cifar10_data()

    datagen = ImageDataGenerator(
        horizontal_flip=True, width_shift_range=0.1,
        height_shift_range=0.1, rotation_range=15, zoom_range=0.1
    )
    datagen.fit(x_tr)

    model = get_model('image')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        datagen.flow(x_tr, y_tr, batch_size=64),
        validation_data=(x_te, y_te),
        epochs=40,
        steps_per_epoch=len(x_tr) // 64,
        callbacks=get_callbacks('image'),
        verbose=1
    )
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Image CNN] Test Accuracy: {acc*100:.2f}%")


# ── Model 2: Text LSTM ────────────────────────────────────────────────────────

def train_text():
    print("\n=== Training Text LSTM (IMDB Sentiment) ===")
    (x_tr, y_tr), (x_te, y_te) = load_imdb_data()

    model = get_model('text')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        x_tr, y_tr,
        validation_data=(x_te, y_te),
        epochs=10,
        batch_size=128,
        callbacks=get_callbacks('text'),
        verbose=1
    )
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Text LSTM] Test Accuracy: {acc*100:.2f}%")


# ── Model 3: Tabular DNN ──────────────────────────────────────────────────────

def train_tabular():
    print("\n=== Training Tabular DNN (Iris) ===")
    (x_tr, y_tr), (x_te, y_te) = load_iris_data()

    model = get_model('tabular')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        x_tr, y_tr,
        validation_data=(x_te, y_te),
        epochs=100,
        batch_size=16,
        callbacks=get_callbacks('tabular'),
        verbose=1
    )
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Tabular DNN] Test Accuracy: {acc*100:.2f}%")


# ── Model 4: Audio 1D-CNN ─────────────────────────────────────────────────────

def train_audio():
    print("\n=== Training Audio 1D-CNN (Synthetic Signals) ===")
    (x_tr, y_tr), (x_te, y_te) = generate_audio_dataset()

    model = get_model('audio')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        x_tr, y_tr,
        validation_data=(x_te, y_te),
        epochs=20,
        batch_size=32,
        callbacks=get_callbacks('audio'),
        verbose=1
    )
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Audio 1D-CNN] Test Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all',
                        help='Model to train: image, text, tabular, audio, or all')
    args = parser.parse_args()
    target = args.model.lower()

    if target in ('tabular', 'all'):
        train_tabular()   # fastest — train first
    if target in ('audio', 'all'):
        train_audio()     # ~2 min
    if target in ('text', 'all'):
        train_text()      # ~10 min
    if target in ('image', 'all'):
        train_image()     # ~20 min

    print("\n[OK] All training complete.")
