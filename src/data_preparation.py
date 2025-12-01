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

if __name__ == "__main__":
    texts, labels = load_all_domains("data")
    print("Total reviews:", len(texts))
    print("Example text:", texts[0][:200])
    print("First 10 labels:", labels[:10])
    print("Number of positive labels:", sum(labels))
    print("Number of negative labels:", len(labels) - sum(labels))
