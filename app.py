import streamlit as st
import joblib
import pandas as pd
import re

# Load model and features
model = joblib.load("spam_model.pkl")
feature_columns = joblib.load("features.pkl")

# Function to convert email text to feature vector
def email_to_features(email_text):
    email_text = email_text.lower()
    words = re.findall(r'\b\w+\b', email_text)

    feature_dict = {word: 0 for word in feature_columns}
    for word in words:
        if word in feature_dict:
            feature_dict[word] += 1

    return pd.DataFrame([feature_dict])

# Streamlit UI
st.title("📧 Spam Email Detection System")
st.write("Enter an email message to check whether it is **SPAM** or **NOT SPAM**.")

email_text = st.text_area("✉️ Enter Email Text")

if st.button("Check Spam"):
    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        email_features = email_to_features(email_text)
        prediction = model.predict(email_features)[0]
        probability = model.predict_proba(email_features)[0][1]

        if prediction == 1:
            st.error(f"🚨 SPAM EMAIL\n\nSpam Probability: {probability:.2f}")
        else:
            st.success(f"✅ NOT SPAM\n\nSpam Probability: {probability:.2f}")
