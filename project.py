import pandas as pd
import numpy as np
import pickle
import cv2
import pytesseract
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ✅ Load dataset (Ensure CSV files exist in "dataset/")
true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")

# ✅ Add labels
true_df["label"] = 1
fake_df["label"] = 0

# ✅ Combine & Shuffle
df = pd.concat([true_df, fake_df]).sample(frac=1, random_state=42)

# ✅ TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["title"] + " " + df["text"])
y = df["label"]

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train multiple models
models = {
    "NaiveBayes": MultinomialNB(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# ✅ Save models
for name, model in models.items():
    model.fit(X_train, y_train)
    pickle.dump(model, open(f"dataset/{name}_model.pkl", "wb"))

# ✅ Save vectorizer
pickle.dump(vectorizer, open("dataset/tfidf_vectorizer.pkl", "wb"))

# ✅ Trusted domains
trusted_domains = {
    "bbc.com": 0.9,
    "nytimes.com": 0.85,
    "reuters.com": 0.88,
    "theguardian.com": 0.86,
}

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh)
    return text.strip()

def classify_news(text, model_name="NaiveBayes", url=""):
    """Predict news as Real or Fake based on selected model."""
    vectorizer = pickle.load(open("dataset/tfidf_vectorizer.pkl", "rb"))
    model = pickle.load(open(f"dataset/{model_name}_model.pkl", "rb"))

    text_vectorized = vectorizer.transform([text])
    probability = model.predict_proba(text_vectorized)[0][1]

    # Adjust probability based on trusted domains
    for domain, weight in trusted_domains.items():
        if domain in url:
            probability = max(probability, weight)

    return "Real News" if probability >= 0.5 else "Fake News", probability

# ✅ Test classification
if __name__ == "__main__":
    sample_text = "Breaking news: Major AI breakthrough!"
    sample_url = "https://www.nytimes.com/2025/03/25/tech/ai-news.html"

    result, prob = classify_news(sample_text, "NaiveBayes", sample_url)
    print(f"🔍 Result: {result} (Confidence: {prob:.2f})")
