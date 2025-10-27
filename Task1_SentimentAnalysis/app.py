import streamlit as st
import joblib
import re
import requests
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

#  Load Model & Vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

#  Your TMDB API Key
TMDB_KEY = "51449b09dc7e1486f4fda013d6066edf"

#  Clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

#  Step1: Convert IMDB → TMDB ID
def imdb_to_tmdb_id(imdb_id):
    url = f"https://api.themoviedb.org/3/find/{imdb_id}?api_key={TMDB_KEY}&external_source=imdb_id"
    r = requests.get(url).json()
    if "movie_results" in r and len(r["movie_results"]) > 0:
        return r["movie_results"][0]["id"]
    return None

#  Step2: Fetch ALL Reviews from TMDB
def get_tmdb_reviews(tmdb_id):
    reviews = []
    page = 1

    while True:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews?api_key={TMDB_KEY}&page={page}"
        data = requests.get(url).json()

        if "results" not in data or len(data["results"]) == 0:
            break

        for rev in data["results"]:
            reviews.append(rev["content"])

        if page >= data.get("total_pages", 1):
            break
        page += 1

    return reviews


#  Streamlit UI
st.set_page_config(page_title="🎬 Movie Sentiment Analyzer", page_icon="🎥")
st.title("🎬 Movie Sentiment Analysis")
st.markdown("Analyze real audience reviews and determine whether they express positive or negative sentiment!")

option = st.radio("Choose Input:", ["✍️ Write a Review", "🔗 Movie Link"])

#  Manual Review
if option == "✍️ Write a Review":
    review = st.text_area("Write your review:")

    if st.button("Analyze"):
        if not review.strip():
            st.warning("⚠️ Please write a review!")
            st.stop()

        clean_r = clean_text(review)
        pred = model.predict(vectorizer.transform([clean_r]))[0]

        if pred == 1:
            sentiment, color = ("🌟 Positive 😊", "#4CAF50")
        else:
            sentiment, color = ("💔 Negative 😞", "#E74C3C")

        st.markdown(
            f"""
            <div style="background-color:{color}; padding:15px; border-radius:10px; margin-top:15px;">
                <h3 style="color:white; text-align:center; font-weight:bold;">
                    {sentiment}
                </h3>
            </div>
            """, unsafe_allow_html=True
        )


#  Movie Link
else:
    link = st.text_input("Paste IMDb Link (example: https://www.imdb.com/title/tt0111161/)")

    if st.button("Analyze Movie"):
        match = re.search(r"(tt\d+)", link)
        if not match:
            st.error("❌ Invalid IMDb link! Example: https://www.imdb.com/title/tt1234567/")
            st.stop()

        imdb_id = match.group(1)
        tmdb_id = imdb_to_tmdb_id(imdb_id)

        if not tmdb_id:
            st.error("❌ Movie not found on TMDB!")
            st.stop()

        reviews = get_tmdb_reviews(tmdb_id)

        if not reviews:
            st.error("❌ No reviews found!")
            st.stop()

        pos = neg = 0
        for r in reviews:
            clean_r = clean_text(r)
            pred = model.predict(vectorizer.transform([clean_r]))[0]
            pos += (pred == 1)
            neg += (pred == 0)

        total = pos + neg
        percent = (pos / total) * 100

        if pos > neg:
            sentiment, color = ("🌟 Overall Positive 😊", "#4CAF50")
        else:
            sentiment, color = ("💔 Overall Negative 😞", "#E74C3C")

        st.markdown(
            f"""
            <div style="background-color:{color}; padding:15px; border-radius:10px; margin-top:15px;">
                <h3 style="color:white; text-align:center; font-weight:bold;">
                    {sentiment}
                </h3>
            </div>
            """, unsafe_allow_html=True
        )


