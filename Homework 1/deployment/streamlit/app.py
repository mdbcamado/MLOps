import streamlit as st
import pandas as pd
import joblib

def main():
    st.title("Fruit Classifier")
    
    # Input features
    mass = st.number_input("Mass", min_value=0.0)
    width = st.number_input("Width", min_value=0.0)
    height = st.number_input("Height", min_value=0.0)
    color_score = st.number_input("Color Score", min_value=0.0, max_value=1.0)
    
    if st.button("Predict"):
        model = joblib.load("models/fruit_classifier.joblib")
        prediction = model.predict([[mass, width, height, color_score]])
        st.write(f"Predicted fruit: {prediction[0]}")

if __name__ == "__main__":
    main()