import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

def main():
    st.title("Fruit Classifier")
    
    # Input for original features
    st.header("Enter Fruit Measurements")
    mass = st.number_input("Mass", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)
    height = st.number_input("Height", min_value=0.0)
    color_score = st.number_input("Color Score", min_value=0.0, max_value=1.0)
    
    # Create a DataFrame for scaling only when the input values are updated
    if mass > 0 and width > 0 and height > 0:
        input_data = pd.DataFrame([[mass, width, height, color_score]], columns=['mass', 'width', 'height', 'color_score'])
        
        # Load the pre-fitted scaler
        scaler = joblib.load("models/scaler.joblib")
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)  # Use transform instead of fit_transform

        # Update the values with scaled data
        mass_scaled, width_scaled, height_scaled, color_score_scaled = scaled_data[0]

        # Show scaled values
        st.header("Scaled Features")
        st.write(f"Scaled Mass: {mass_scaled:.2f}")
        st.write(f"Scaled Width: {width_scaled:.2f}")
        st.write(f"Scaled Height: {height_scaled:.2f}")
        st.write(f"Scaled Color Score: {color_score_scaled:.2f}")
        

        area = width_scaled * height_scaled
        density = mass_scaled / area 
        aspect_ratio = height_scaled / width_scaled
        
        
        # Show engineered features
        st.header("Calculated Features")
        st.write(f"Area: {area:.2f}")
        st.write(f"Density: {density:.2f}")
        st.write(f"Aspect Ratio: {aspect_ratio:.2f}")
    
        if st.button("Predict"):
            # Prepare all features (scaled + engineered features)
            features = [mass_scaled, width_scaled, height_scaled, color_score_scaled, area, density, aspect_ratio]
            
            # Load model and predict
            model = joblib.load("models/fruit_classifier.joblib")
            prediction = model.predict([features])
            
            # Map prediction to fruit name
            fruit_map = {1: 'apple', 2: 'mandarin', 3: 'orange', 4: 'lemon'}
            fruit_name = fruit_map[prediction[0]]
            
            st.success(f"Predicted fruit: {fruit_name}")

if __name__ == "__main__":
    main()
