"""
retrain_text.py — Retrain the Bidirectional LSTM on IMDB sequences.

Matches the saved model architecture exactly:
  Embedding(10000, 64) → BiLSTM(128, return_sequences=True) → BiLSTM(64) → Dense(64) → Dense(2)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks

VOCAB_SIZE   = 10000
SEQUENCE_LEN = 200
SAVE_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_text.keras')


def build_text_lstm():
    inp = tf.keras.Input(shape=(SEQUENCE_LEN,), name='input_layer')
    x   = layers.Embedding(VOCAB_SIZE, 64, input_length=SEQUENCE_LEN)(inp)
    x   = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x   = layers.Bidirectional(layers.LSTM(32))(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(inp, out, name='TextLSTM')


def main():
    print("Loading IMDB dataset...")
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)

    print("Padding sequences...")
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    x_tr = pad_sequences(x_tr, maxlen=SEQUENCE_LEN, padding='pre', truncating='pre')
    x_te = pad_sequences(x_te, maxlen=SEQUENCE_LEN, padding='pre', truncating='pre')

    y_tr = tf.keras.utils.to_categorical(y_tr, 2)
    y_te = tf.keras.utils.to_categorical(y_te, 2)

    print(f"Train: {x_tr.shape}, Test: {x_te.shape}")

    model = build_text_lstm()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(SAVE_PATH, save_best_only=True,
                                  monitor='val_accuracy', verbose=1),
        callbacks.EarlyStopping(patience=3, restore_best_weights=True,
                                monitor='val_accuracy', verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2,
                                    min_lr=1e-6, verbose=1),
    ]

    print("\nTraining BiLSTM...")
    model.fit(
        x_tr, y_tr,
        validation_data=(x_te, y_te),
        epochs=10,
        batch_size=128,
        callbacks=cbs,
        verbose=1
    )

    _, acc = model.evaluate(x_te, y_te, verbose=0)
    print(f"\n[TextLSTM] Final Test Accuracy: {acc*100:.2f}%")
    print(f"[TextLSTM] Saved to: {SAVE_PATH}")


if __name__ == '__main__':
    main()
