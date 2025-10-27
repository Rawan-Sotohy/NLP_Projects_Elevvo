# 🎯 Task 1 — Sentiment Analysis on Movie Reviews  

## 📘 Overview

The project focuses on classifying reviews as **Positive** or **Negative**, applying text preprocessing, feature extraction, and model training.  
Finally, the model will be deployed in an interactive web app that allows users to enter a movie review (or movie link) and get instant sentiment predictions.

---
## Dataset
- **Source:** [IMDb Dataset of 50k Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)  
- **Columns:**
  - `review` → text of the review  
  - `sentiment` → Positive / Negative  

---
## 🧰 Tech Stack

- **Python**
- **pandas**, **numpy**
- **nltk**, **spacy**
- **scikit-learn**
- **matplotlib**, **seaborn**, **wordcloud**
- **TMDB/IMDb API Integration**
- **streamlit**

---

## 🧠 Project Pipeline

### **1️⃣ Data Loading & Exploration**
- Import IMDb dataset (50k reviews)
- Explore data shape, missing values, and sentiment distribution

### **2️⃣ Text Preprocessing**
- Convert to lowercase  
- Remove punctuation & stopwords   

### **3️⃣ Text Representation**
- Transform text into numeric form using:
  - **CountVectorizer**
  - **TF-IDF Vectorizer**

### **4️⃣ Model Training & Evaluation**
- Train classifiers:
  - Logistic Regression  
  - Naive Bayes 
- Evaluate using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

### **5️⃣ Visualization**
- Generate **WordClouds** for positive & negative reviews  
- Visualize word frequency and sentiment distribution  

### **6️⃣ Deployment**
An interactive **Streamlit App** allows two input options:
1. **Write a Review:** Type your review directly  
2. **Movie Link:** Provide a movie link to fetch a review  

---

## 🎬 Demo

Try the **movie review sentiment classifier** live through the Streamlit app:

[🌐 Open the App Here](https://moviesentimentanalysis2025.streamlit.app/)

---
The app predicts the review sentiment and displays the result in real-time.
