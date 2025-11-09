# Fake News Detection Web App 
import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    return text

# Streamlit UI setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.markdown('<h1 >📰 Fake News Detection App</h1>', unsafe_allow_html=True)
st.markdown("### Detect if a news article is **Fake** or **Real** using AI 🤖")

# Custom CSS for badges and button
st.markdown(
    """
    <style>
    .fake {background-color:#ff4b4b; color:white; padding:10px; border-radius:10px;}
    .real {background-color:#4caf50; color:white; padding:10px; border-radius:10px;}
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
        height: 50px;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True
)

# Form to handle input and button
with st.form(key='analyze_form'):
    user_input = st.text_area("Enter News Text:", height=200)
    submit_button = st.form_submit_button("Analyze")
    
    if submit_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            clean_input = clean_text(user_input)
            input_vec = vectorizer.transform([clean_input])
            prediction = model.predict(input_vec)[0]
            
            # Display result with colors only (no probabilities)
            if prediction == 1:
                st.markdown(f'<p class="real">✅ This news is REAL</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="fake">🚨 This news is FAKE</p>', unsafe_allow_html=True)
