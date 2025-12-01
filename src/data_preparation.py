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


if __name__ == "__main__":
    sample = load_reviews("data/dvd_positive.review")  # đổi thành file bạn có
    print("Loaded reviews:", len(sample))
    print("First review sample:\n", sample[0][:300])
