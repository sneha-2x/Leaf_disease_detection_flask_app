# 🌿 Leaf Disease Detection 🚀

A deep learning-powered Flask web application that detects plant leaf diseases from images using a trained Convolutional Neural Network (CNN) model.

## 📌 Features

- 🌱 Upload a leaf image via web interface
- 🔍 Classifies the image into healthy/diseased categories
- 📊 Displays prediction result with confidence score
- 🧠 Powered by a trained deep learning model
- 🌐 Simple Flask backend with HTML frontend

---

## 🧠 Model Info

- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Trained On**: Custom dataset of leaf images with various plant diseases
- **Classes**: (e.g., Healthy, Powdery Mildew, Rust, Blight...) – *(update with actual classes used)*

---

## 📁 Project Structure

leaf_disease_detection/
│
├── model/ # Saved CNN model (.h5)
│ └── model.h5
│
├── static/ # Static assets like CSS/images
│
├── templates/ # HTML templates (index.html)
│
├── app.py # Flask app
├── requirements.txt # Python dependencies
└── README.md # Project documentation
