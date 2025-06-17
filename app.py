import streamlit as st
import os
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import base64

# Load trained model
model = load_model("violence_detection_model.h5")

# Constants
IMG_SIZE = 128
MAX_FRAMES = 20
CLASS_NAMES = ["NonViolence", "Violence"]

# Set Streamlit page configuration
st.set_page_config(page_title="Violence Detection App", page_icon="⚠", layout="wide")

# Convert background image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"

# Background CSS styling
bg_image = get_base64_image("background.png")
st.markdown(f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), url("{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .section {{
        background-color: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
        margin: 20px 0;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span, .stMarkdown {{
        color: white !important;
    }}
    .stButton>button {{
        background-color: rgba(255, 75, 75, 0.6);
        color: white;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
    }}
    .stButton>button:hover {{
        background-color: rgba(255, 75, 75, 0.9);
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# Extract frames from video
def extract_frames(video_path, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    cap.release()
    return np.array(frames)

# Preprocess frames
def preprocess_frames(frames):
    return frames.astype("float32") / 255.0

# Predict violence
def predict(frames):
    preds = []
    for frame in frames:
        img = img_to_array(frame)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img, verbose=0)
        preds.append(pred[0][0])
    avg_pred = np.mean(preds)
    return "Violence" if avg_pred > 0.5 else "NonViolence", avg_pred, preds

# UI - Header
st.markdown('<div class="section"><h1>⚠ Violence Detection in Video</h1></div>', unsafe_allow_html=True)

# Upload Section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📤 Upload a Video")
video_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
if video_file:
    col1, col2 = st.columns([1, 1])

    # Column 1 - Video Preview
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("🎥 Video Preview")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

    # Column 2 - Prediction
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.header("🔍 Processing & Prediction")
        with st.spinner("Analyzing video..."):
            frames = extract_frames(tmp_path)
            preprocessed = preprocess_frames(frames)
            label, confidence, confidence_scores = predict(preprocessed)
        st.success(f"✅ Prediction: *{label}* ({confidence:.2f} confidence)")
        st.markdown('</div>', unsafe_allow_html=True)

    # Graphs Section
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.header("📊 Visualization")
    graph_option = st.selectbox(
        "Choose a graph to visualize the results:",
        ["Confidence Score Distribution", "Frame-wise Prediction", "Classification Summary", "Confidence Score Histogram"]
    )

    frame_numbers = np.arange(1, len(confidence_scores) + 1)
    fig, ax = plt.subplots(figsize=(5, 3))
    if graph_option == "Confidence Score Distribution":
        ax.plot(frame_numbers, confidence_scores, marker='o', linestyle='-', color='red')
        ax.set_title("Confidence Score Across Frames")
    elif graph_option == "Frame-wise Prediction":
        ax.scatter(frame_numbers, confidence_scores, c=['red' if x > 0.5 else 'blue' for x in confidence_scores])
        ax.axhline(y=0.5, color='gray', linestyle='--')
        ax.set_title("Frame-wise Prediction")
    elif graph_option == "Classification Summary":
        labels = ["Violence", "NonViolence"]
        counts = [sum(x > 0.5 for x in confidence_scores), sum(x <= 0.5 for x in confidence_scores)]
        ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['red', 'blue'])
        ax.set_title("Overall Classification Summary")
    elif graph_option == "Confidence Score Histogram":
        ax.hist(confidence_scores, bins=10, color='purple', alpha=0.7)
        ax.set_title("Distribution of Confidence Scores")
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Disclaimer Section
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("⚠ Disclaimer")
st.write("This AI model is not 100% accurate and should not be used for legal or security decisions. Always verify results manually.")
st.markdown('</div>', unsafe_allow_html=True)