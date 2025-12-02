from flask import Flask, render_template, request
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("saved_model/sentiment_model.h5")

with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 120

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None

    if request.method == "POST":
        text = request.form["review"].lower()
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

        prob = model.predict(padded)[0][0]
        result = "Positive" if prob >= 0.5 else "Negative"

    conf = prob * 100 if prob is not None else None
    # return render_template("index.html", result=result, confidence=conf)
    return render_template("index.html", result=result, confidence=conf, review_text=text if request.method=="POST" else "") #to not clear typed text 



if __name__ == "__main__":
    app.run(debug=True)
