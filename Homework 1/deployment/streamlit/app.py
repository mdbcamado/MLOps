import streamlit as st
import pandas as pd
import joblib
import numpy as np

def main():
    st.title("Fruit Classifier")
    
    # Input for original features
    st.header("Enter Fruit Measurements")
    mass = st.number_input("Mass", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)
    height = st.number_input("Height", min_value=0.0)
    color_score = st.number_input("Color Score", min_value=0.0, max_value=1.0)
    
    # Calculate engineered features
    if width > 0 and height > 0:  # Avoid division by zero
        area = width * height
        density = mass / area if mass > 0 else 0
        aspect_ratio = height / width
    else:
        area = density = aspect_ratio = 0
    
    # Show engineered features
    st.header("Calculated Features")
    st.write(f"Area: {area:.2f}")
    st.write(f"Density: {density:.2f}")
    st.write(f"Aspect Ratio: {aspect_ratio:.2f}")
    
    if st.button("Predict"):
        # Prepare all features
        features = [mass, width, height, color_score, area, density, aspect_ratio]
        
        # Load model and predict
        model = joblib.load("models/fruit_classifier.joblib")
        prediction = model.predict([features])
        
        # Map prediction to fruit name
        fruit_map = {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}
        fruit_name = fruit_map[prediction[0]]
        
        st.success(f"Predicted fruit: {fruit_name}")

if __name__ == "__main__":
    main()