import streamlit as st
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from project import classify_news, extract_text_from_image

# 🎨 Custom Styling
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.markdown("""
    <style>
    .stTitle {text-align: center; color: #ff4b4b;}
    .stRadio label {font-size: 16px !important;}
    .stButton button {background-color: #ff4b4b; color: white; font-size: 16px;}
    .stSelectbox div {font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# 📰 AI-Based Fake News Detector
st.title("🚀 AI-Powered Fake News Detector")
st.markdown("Detect misinformation using AI-powered models. Upload text or an image to verify its authenticity.")

# 📌 Sidebar Options
st.sidebar.header("🔍 Settings")
model_choice = st.sidebar.selectbox("Choose a Model:", ["NaiveBayes", "LogisticRegression", "RandomForest"])
option = st.sidebar.radio("Select Input Type:", ["Text", "Image"])

# 📑 Text-Based Input
if option == "Text":
    user_input = st.text_area("Enter news text here:", height=150)
    if st.button("Analyze Text", use_container_width=True):
        if user_input.strip():
            result, prob = classify_news(user_input, model_choice)
            st.success(f"✅ Prediction: {result} (Confidence: {prob:.2f})")

            # 📊 Pie Chart
            fig, ax = plt.subplots()
            labels = ["Real News", "Fake News"]
            colors = ["green", "red"]
            values = [prob, 1 - prob]
            ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Please enter some text.")

# 📷 Image-Based Input
elif option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_path = "temp_image.png"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        extracted_text = extract_text_from_image(file_path)
        st.text_area("📜 Extracted Text:", extracted_text, height=150, disabled=True)

        if extracted_text.strip() and st.button("Analyze Image", use_container_width=True):
            result, prob = classify_news(extracted_text, model_choice)
            st.success(f"✅ Prediction: {result} (Confidence: {prob:.2f})")

            # 📊 Pie Chart
            fig, ax = plt.subplots()
            labels = ["Real News", "Fake News"]
            colors = ["green", "red"]
            values = [prob, 1 - prob]
            ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
            ax.axis("equal")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload an image.")

st.markdown("---")
st.caption("🛠️ Built with Streamlit | AI-Powered Fake News Detector")
