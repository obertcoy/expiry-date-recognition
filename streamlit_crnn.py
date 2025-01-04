import streamlit as st
from PIL import Image
import pytesseract
import tensorflow as tf
import numpy as np
import pickle
from crnn import load_saved_model, get_model_output, nums_to_string
from preprocess import crnn_preprocess_image, tesseract_preprocess_image, clean_text, traditional_preprocess
from utils import format_to_datestring

def load_crnn_model(input_shape=(64, 200, 1)):
    
    try:
        model = load_saved_model(input_shape=input_shape)
        return model
    
    except Exception as e:
        st.error(f"Error loading CRNN model: {e}")
        return None


def main():
    st.title("Expiry Date Recognition")

    uploaded_file = st.file_uploader("Upload Image of Expiry Date", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Recognize Text"):
                
            preprocessed_image = crnn_preprocess_image(image)
            
            # st.write(f'Preprocessed image shape: {preprocessed_image.shape}')
            
            crnn_model = load_crnn_model(input_shape= preprocessed_image.shape)

            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

            Y_pred = crnn_model.predict(preprocessed_image)
                            
            output = get_model_output(Y_pred)
            
            result = format_to_datestring(output)

            st.markdown(
                f"""
                <div style="text-align: center; font-size: 24px; font-weight: bold;">
                    Detected Text using CRNN
                </div>
                <div style="text-align: center; font-size: 20px;">
                    {result}
                </div>
                """,
                unsafe_allow_html=True
            )    
                
                    
if __name__ == "__main__":
    main()
