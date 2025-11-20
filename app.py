from flask import Flask, request, render_template, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# -------------------------
# NLTK setup
# -------------------------
nltk.data.path.append("nltk_data")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------
# CLEANING FUNCTION
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text).strip()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return " ".join(tokens)

# -------------------------
# LOAD MODELS + VECTORIZER
# -------------------------
vectorizer = joblib.load("vectorizer.pkl")
log_reg = joblib.load("logistic_regression_model.pkl")
xgb = joblib.load("xgb.pkl")

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_sentiment_single(tweet, model="logistic"):
    cleaned = clean_text(tweet)
    vec = vectorizer.transform([cleaned])
    if model == "logistic":
        pred_code = log_reg.predict(vec)[0]
    else:
        pred_code = xgb.predict(vec)[0]
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map.get(pred_code, "Unknown")

def predict_sentiment_batch(tweets, model="logistic"):
    cleaned_tweets = [clean_text(tweet) for tweet in tweets]
    vec = vectorizer.transform(cleaned_tweets)
    if model == "logistic":
        pred_codes = log_reg.predict(vec)
    else:
        pred_codes = xgb.predict(vec)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return [label_map.get(code, "Unknown") for code in pred_codes]

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# Main page
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# Batch prediction endpoint
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()
    tweets = data.get("tweets", [])
    model = data.get("model", "logistic")
    if not tweets:
        return jsonify({"predictions": []})
    predictions = predict_sentiment_batch(tweets, model)
    # Return results as array of {tweet, predicted_sentiment}
    results = [{"tweet": t, "predicted_sentiment": p} for t, p in zip(tweets, predictions)]
    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(debug=True)
