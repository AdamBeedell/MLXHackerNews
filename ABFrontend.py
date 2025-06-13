import requests
import streamlit as st
import time
import random


st.title("🧠 Hacker News Upvote Predictor")

title = st.text_input("Title (required)")
url = st.text_input("URL (optional)")
user = st.text_input("Username (optional)")


if st.button("Submit"):
    if not title.strip():
        st.error("Title is required.")
    else:
        try:
            response = requests.post(
                "http://localhost:8000/api/predict",
                json={"title": title, "url": url, "user": user}
            )
            response.raise_for_status()
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
            st.write(f"Title: {result['title']}")
            st.write(f"User: {result['user']}")
            st.write(f"Domain: {result['domain']}")
            st.balloons()
        except requests.exceptions.RequestException as e:
            st.error(f"Error contacting prediction API: {e}")
    
    


