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

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\shrey\Downloads\Tesseract-OCR\tessdata"

# Define the OCR extraction function directly in the Streamlit app
def extract_text_from_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Image not found or could not be loaded at path: {image_path}")

    # Convert the image to RGB (from BGR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use pytesseract to do OCR on the image
    extracted_text = pytesseract.image_to_string(rgb_image)

    return extracted_text

# Streamlit app
st.title("Image Text Extractor with OCR")
st.write("Upload an image, and the app will extract the text using OCR.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save the image to a temporary file
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text using the defined OCR function
    extracted_text = extract_text_from_image("temp_image.png")

    # Display the extracted text
    st.subheader("Extracted Text:")
    st.write(extracted_text)

