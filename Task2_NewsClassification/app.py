from flask import Flask, render_template, request, jsonify
import joblib, os, re
import webbrowser
from threading import Timer


# Paths to models
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

# Load artifacts
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# Simple text cleaning
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)       # keep letters only
    text = text.lower()
    return ' '.join(text.split())

# Initialize Flask
app = Flask(__name__)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Predict route (AJAX POST)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error":"No text provided"}), 400

    text = data['text']
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    pred = model.predict(X)[0]

    # Try to inverse transform if encoded
    try:
        label = le.inverse_transform([pred])[0]
    except:
        label = str(pred)

    return jsonify({"category": label})


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
