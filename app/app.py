from flask import Flask, render_template, request
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model("saved_model/sentiment_model.h5") #loading trained model

with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f) #loading tokenizer

MAX_LEN = 120

app = Flask(__name__) #creating web server (for handling requests)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    prob = None

    if request.method == "POST":
        text = request.form["review"].lower()
        seq = tokenizer.texts_to_sequences([text]) #making sequense from text to get tokens
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post") # making seq with fixed length 


        # DEBUG prints
        print("DEBUG — tokenizer OOV token = ", tokenizer.oov_token)
        print("DEBUG — tokenizer OOV index = ", tokenizer.word_index.get(tokenizer.oov_token))
        print("DEBUG — tokens = ", seq[0])
        print("DEBUG — text = ", text)

        ## turning tokens -> to words
        #reverse_dict = {v: k for k, v in tokenizer.word_index.items()}

        ## all input words
        #words = text.split()

        ## add words with tok=1 into OOV_words
        #oov_words = []
        #for w, tok in zip(words, seq[0]):
        #    if tok == 1:
        #        oov_words.append(w)


        prob = model.predict(padded)[0][0] # prediction of model


        raw_prob = float(prob)
        tokens = seq[0]                  # tokens list
        seq_len = len(tokens)            # length
        #oov_count = tokens.count(1)      # 1 = OOV-token


        # index of OOV-token 
        oov_index = tokenizer.word_index.get(tokenizer.oov_token)

        #restored_words = tokenizer.sequences_to_texts(seq)[0].split() #restoring words as tokenizer works with them
        #oov_words = [w for w, tok in zip(restored_words, tokens) if tok == oov_index]

        words = text.split()
        #oov_words = []
        #oov_words = [w for w, tok in zip(words, tokens) if tok == oov_index]
        oov_words = []
        for w, tok in zip(words, tokens):
            print("w = ", w)
            print("tok = ", tok)
            print("oov_index = ", oov_index)
            if tok == oov_index:
                oov_words.append(w)

        #for w, tok in zip(words, tokens):
        #    if tok == oov_index:
        #        oov_words.append(w)


        result = "Positive" if prob >= 0.5 else "Negative"


        debug = {
            "tokens": tokens,
            "seq_len": seq_len,
            #"oov_count": oov_count,
            "raw_prob": raw_prob,
            "oov_words": oov_words,
        }

    conf = prob * 100 if prob is not None else None
    # return render_template("index.html", result=result, confidence=conf)
    # return render_template("index.html", result=result, confidence=conf, review_text=text if request.method=="POST" else "") #to not clear typed text 
    return render_template(
        "index.html",
        result=result,
        confidence=conf,
        review_text=text if request.method=="POST" else "",
        debug=debug if request.method=="POST" else None
    )


if __name__ == "__main__":
    app.run(debug=True)
