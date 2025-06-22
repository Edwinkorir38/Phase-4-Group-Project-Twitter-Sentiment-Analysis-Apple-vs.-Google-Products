import streamlit as st
import pickle
import re

# Load models
with open(r"env/logistic_regression_model.pkl", "rb") as f:
    binary_model = pickle.load(f)

with open(r"env/xgb.pkl", "rb") as f:
    multiclass_model = pickle.load(f)

# Cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    return text

# Streamlit UI
st.title("Twitter Sentiment Classifier")

tweet = st.text_input("Enter Tweet:")
model_type = st.radio("Select Model Type", ["Binary", "Multiclass"])

if st.button("Predict"):
    cleaned = clean_text(tweet)

    if model_type == "Binary":
        pred = binary_model.predict([cleaned])[0]
        st.success(f"Predicted Sentiment (Binary): {'Positive' if pred==1 else 'Negative'}")

    else:
        pred = multiclass_model.predict([cleaned])[0]
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust based on your labels
        st.success(f"Predicted Sentiment (Multiclass): {label_map[pred]}")