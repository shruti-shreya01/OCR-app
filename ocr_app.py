# import streamlit as st
# import cv2
# import pytesseract
# from PIL import Image
# import numpy as np
# import tempfile
# import os

# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update if necessary

# st.title("Image Text Extractor with OCR")
# st.write("Upload an image, and the app will extract the text using OCR (Tesseract).")

# uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#         temp_file.write(uploaded_file.read())
#         image_path = temp_file.name

#     def extract_text_from_image(image_path):
#         image = cv2.imread(image_path)
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         extracted_text = pytesseract.image_to_string(rgb_image)
#         return extracted_text

#     extracted_text = extract_text_from_image(image_path)
#     st.subheader("Extracted Text:")
#     st.write(extracted_text)


import streamlit as st
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Load the OCR model from the pickle file
def load_pickle_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load the OCR extraction function
extract_text_from_image = load_pickle_model('ocr_model.pkl')

# Streamlit app
st.title("Image Text Extraction using OCR")

# Upload an image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the image to a temporary location
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text using the OCR model
    extracted_text = extract_text_from_image("temp_image.png")

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)
