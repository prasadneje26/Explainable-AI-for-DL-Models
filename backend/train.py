"""
train.py — Proper training for all 4 models with data augmentation.

Usage:
    python train.py              # all 4
    python train.py --model image / text / tabular / audio
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import get_model, MODEL_SAVE_PATHS
from data import load_mnist, load_imdb_bow, load_iris_data, generate_audio_dataset


def cbs(name):
    return [
        callbacks.ModelCheckpoint(MODEL_SAVE_PATHS[name], save_best_only=True,
                                  monitor='val_accuracy', verbose=1),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                monitor='val_accuracy', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=3, min_lr=1e-6, verbose=1)
    ]


def train_image():
    print("\n=== Image CNN — MNIST ===")
    (x_tr, y_tr), (x_te, y_te) = load_mnist()

    # Data augmentation: slight rotation and shift to improve generalization
    aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                             height_shift_range=0.1, zoom_range=0.1)
    aug.fit(x_tr)

    m = get_model('image')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit(aug.flow(x_tr, y_tr, batch_size=128),
          validation_data=(x_te, y_te),
          epochs=15, steps_per_epoch=len(x_tr)//128,
          callbacks=cbs('image'), verbose=1)

    _, acc = m.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Image CNN] Final Test Accuracy: {acc*100:.2f}%")


def train_text():
    print("\n=== Text DNN — IMDB ===")
    (x_tr, y_tr), (x_te, y_te) = load_imdb_bow()

    m = get_model('text')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit(x_tr, y_tr, validation_data=(x_te, y_te),
          epochs=20, batch_size=256, callbacks=cbs('text'), verbose=1)

    _, acc = m.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Text DNN] Final Test Accuracy: {acc*100:.2f}%")


def train_tabular():
    print("\n=== Tabular DNN — Iris ===")
    (x_tr, y_tr), (x_te, y_te) = load_iris_data()

    m = get_model('tabular')
    m.compile(optimizer=tf.keras.optimizers.Adam(5e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit(x_tr, y_tr, validation_data=(x_te, y_te),
          epochs=200, batch_size=8, callbacks=cbs('tabular'), verbose=1)

    _, acc = m.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Tabular DNN] Final Test Accuracy: {acc*100:.2f}%")


def train_audio():
    print("\n=== Audio 1D-CNN — Synthetic Signals ===")
    (x_tr, y_tr), (x_te, y_te) = generate_audio_dataset()

    m = get_model('audio')
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()

    m.fit(x_tr, y_tr, validation_data=(x_te, y_te),
          epochs=25, batch_size=32, callbacks=cbs('audio'), verbose=1)

    _, acc = m.evaluate(x_te, y_te, verbose=0)
    print(f"\n[Audio 1D-CNN] Final Test Accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all')
    t = parser.parse_args().model.lower()

    if t in ('tabular', 'all'): train_tabular()
    if t in ('audio',   'all'): train_audio()
    if t in ('text',    'all'): train_text()
    if t in ('image',   'all'): train_image()

    print("\n[OK] All training complete.")
