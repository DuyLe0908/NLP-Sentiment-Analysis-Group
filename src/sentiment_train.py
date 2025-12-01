import os
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.callbacks import EarlyStopping

from data_preparation import prepare_data


def main():
    print("Loading preprocessed dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    print("Done loading data.")
    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", len(y_train))
    print("  X_val:  ", X_val.shape, "y_val:", len(y_val))
    print("  X_test: ", X_test.shape, "y_test:", len(y_test))


if __name__ == "__main__":
    main()