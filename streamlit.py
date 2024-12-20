import streamlit as st
from PIL import Image
import pytesseract
import tensorflow as tf
import numpy as np
import pickle
from crnn import load_saved_model, get_model_output
from preprocess import crnn_preprocess_image, tesseract_preprocess_image, clean_text, traditional_preprocess
from utils import format_to_datestring


SVM_MODEL = 'svm-rbf.pkl'
RANDOM_FOREST_MODEL = 'random-forest.pkl'


# Load CRNN Model
def load_crnn_model(input_shape=(64, 200, 1)):
    
    try:
        model = load_saved_model(input_shape=input_shape)
        return model
    
    except Exception as e:
        st.error(f"Error loading CRNN model: {e}")
        return None


def main():
    st.title("Expiry Date Recognition")

    model_choice = st.selectbox("Select Model", ("Tesseract", "CRNN"))

    uploaded_file = st.file_uploader("Upload Image of Expiry Date", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Recognize Text"):
            if model_choice == "Tesseract":
                
                preprocessed_image = tesseract_preprocess_image(image) 
                                
                Y_pred = pytesseract.image_to_string(image, config='--psm 6')
                
                output = clean_text(Y_pred)
                
                result = format_to_datestring(output)
                
                st.write("### Detected Text using Tesseract")
                st.write(result)

            elif model_choice == "CRNN":
                
                preprocessed_image = crnn_preprocess_image(image)
                
                print(f'Image shape: {preprocessed_image.shape}')
                
                crnn_model = load_crnn_model(input_shape= preprocessed_image.shape)

                Y_pred = crnn_model.predict(preprocessed_image)
                
                output = get_model_output(Y_pred)
                
                result = format_to_datestring(output)

                st.write("### Detected Text using CRNN")
                st.write(result)

            elif model_choice == 'SVM':
                
                processed_images = traditional_preprocess(preprocessed_image)
                
                with open(SVM_MODEL, 'rb') as file:
                    model = pickle.load(file)
                    
                predictions = model.predict(processed_images)
                
                st.write("### Detected Text using SVM")
                st.write("".join(predictions))
                
            elif model_choice == 'random-forest':
                
                processed_images = traditional_preprocess(preprocessed_image)
                
                with open(RANDOM_FOREST_MODEL, 'rb') as file:
                    model = pickle.load(file)
                    
                predictions = model.predict(processed_images)
                
                st.write("### Detected Text using Random Forest")
                st.write("".join(predictions))
                
                    
if __name__ == "__main__":
    main()
