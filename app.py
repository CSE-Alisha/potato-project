import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# Load the saved models
feature_extractor = tf.keras.models.load_model('feature_extractor.h5')
rf_classifier = joblib.load('rf_classifier.pkl')

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Define image preprocessing function
def preprocess_image(image):
    # Resize image to the required input size for DenseNet
    image = image.resize((224, 224))
    # Convert image to array and scale to [0, 1] range
    image_array = np.array(image) / 255.0
    # Expand dimensions to create a batch of 1
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define prediction function
def predict(image):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Extract features
    features = feature_extractor.predict(preprocessed_image)
    # Predict with Random Forest classifier
    prediction = rf_classifier.predict(features)
    # Get the class label
    predicted_class = class_names[prediction[0]]
    return predicted_class

# Streamlit App Interface
st.title("Potato Disease Classification")
st.write("Upload a potato leaf image to classify it as Early Blight, Late Blight, or Healthy.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict and display the result
    st.write("Classifying...")
    predicted_class = predict(image)
    st.write(f"Prediction: **{predicted_class}**")
