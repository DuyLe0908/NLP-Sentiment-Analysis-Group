import os
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.callbacks import EarlyStopping

from data_preparation import prepare_data


VOCAB_SIZE = 20000
MAX_LEN = 120


def main():
    print("Loading preprocessed dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    print("Done loading data.")
    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", len(y_train))
    print("  X_val:  ", X_val.shape, "y_val:", len(y_val))
    print("  X_test: ", X_test.shape, "y_test:", len(y_test))


def build_model():
    model = Sequential([
        Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model



if __name__ == "__main__":
    print("Loading preprocessed dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    print("Done loading data.")

    print("Building model...")
    model = build_model()
    model.summary()