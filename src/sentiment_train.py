import os
import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences

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

def predict_sentence(model, tokenizer, sentence):
    # Preprocessing is simple: lowercase
    text = sentence.lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Positive" if prob >= 0.5 else "Negative"
    return label, float(prob)



if __name__ == "__main__":
    print("== Training pipeline ==")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    model = build_model()
    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, np.array(y_train),
        epochs=5,
        batch_size=128,
        validation_data=(X_val, np.array(y_val)),
        callbacks=[early_stop],
        verbose=2
    )

    loss, acc = model.evaluate(X_test, np.array(y_test), verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    # Save model
    os.makedirs("saved_model", exist_ok=True)
    model_path = os.path.join("saved_model", "sentiment_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Reload tokenizer to test prediction
    with open("saved_model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    print("\n=== Quick prediction test ===")
    s1 = "I really love this product, it works perfectly!"
    s2 = "This is the worst purchase I have ever made."

    print("Sentence 1:", predict_sentence(model, tokenizer, s1))
    print("Sentence 2:", predict_sentence(model, tokenizer, s2))