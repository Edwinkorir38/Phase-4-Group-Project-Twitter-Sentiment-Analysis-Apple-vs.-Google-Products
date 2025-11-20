import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import nltk

nltk.data.path.append("nltk_data")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------
# PREPROCESSING
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = text.strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_pickle("Data/cleaned_dataset.pkl")  # Load Pickle to preserve tweet_text as list

# Option 1: Use tokenized tweet_text and join into string
df["cleaned"] = df["tweet_text"].apply(lambda x: clean_text(" ".join(x)))

# Option 2: Or use the lemmatized_review string directly
# df["cleaned"] = df["lemmatized_review"].apply(clean_text)

# Define features and target
X = df["cleaned"]
y = df["emotion_code"]  # Use emotion_code as target for classification

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# VECTORIZER
# -------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

joblib.dump(vectorizer, "vectorizer.pkl")

# -------------------------
# LOGISTIC REGRESSION
# -------------------------
log_reg = LogisticRegression(max_iter=2000)
log_reg.fit(X_train_vec, y_train)

joblib.dump(log_reg, "logistic_regression_model.pkl")

# -------------------------
# XGBOOST
# -------------------------
xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)

xgb.fit(X_train_vec, y_train)
joblib.dump(xgb, "xgb.pkl")

print("Models + vectorizer saved successfully!")
