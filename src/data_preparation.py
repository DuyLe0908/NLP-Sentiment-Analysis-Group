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


if __name__ == "__main__":
    raw = "This Product is AMAZING!!! :)   10/10, would buy again..."
    cleaned = clean_text(raw)
    print("RAW:     ", raw)
    print("CLEANED: ", cleaned)
