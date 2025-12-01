import os
import re
import random
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_reviews(path):
    reviews = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    parts = text.split("<review_text>")
    for p in parts[1:]:
        content = p.split("</review_text>")[0].strip()
        if len(content) > 0:
            reviews.append(content)
    return reviews

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # remove strange characters, punctuation
    text = re.sub(r"\s+", " ", text).strip()   # combine multiple spaces into 1
    return text

def load_all_domains(base_path="data"):
    domains = [
        "book",
        "dvd",
        "electronics",
        "kitchen"
    ]

    all_texts = []
    all_labels = []

    for domain in domains:
        pos_path = os.path.join(base_path, f"{domain}_positive.review")
        neg_path = os.path.join(base_path, f"{domain}_negative.review")

        pos_reviews = load_reviews(pos_path)
        neg_reviews = load_reviews(neg_path)

        # clean text
        pos_reviews = [clean_text(r) for r in pos_reviews]
        neg_reviews = [clean_text(r) for r in neg_reviews]

        all_texts.extend(pos_reviews)
        all_labels.extend([1] * len(pos_reviews))

        all_texts.extend(neg_reviews)
        all_labels.extend([0] * len(neg_reviews))

    return all_texts, all_labels

def prepare_data():
    print("Loading dataset...")
    texts, labels = load_all_domains("data")

    # bỏ review quá ngắn
    texts2 = []
    labels2 = []
    for t, l in zip(texts, labels):
        if len(t.split()) > 3:
            texts2.append(t)
            labels2.append(l)

    texts = texts2
    labels = labels2

    print(f"Total reviews after filtering: {len(texts)}")

    # shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    # tokenizer
    tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    # save tokenizer
    os.makedirs("saved_model", exist_ok=True)
    with open("saved_model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # sequences + pad
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=120, padding="post", truncating="post")
    y = list(labels)

    # split train / val / test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", len(y_train))
    print("  X_val:  ", X_val.shape, "y_val:", len(y_val))
    print("  X_test: ", X_test.shape, "y_test:", len(y_test))

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    print("Running test for prepare_data() ...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    print("Data preparation finished.")