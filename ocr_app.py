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
import numpy as np
import tempfile

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update if necessary for local execution

# Streamlit app
st.title("Image Text Extractor with OCR")

st.write("Upload an image, and the app will extract the text using OCR (Tesseract).")

# Image uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image in Streamlit
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert PIL image to NumPy array (OpenCV format)
    img_array = np.array(image)

    # Convert the image to BGR (required by OpenCV)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Extract text from the image using Tesseract OCR
    def extract_text_from_image(image_bgr):
        # Use pytesseract to perform OCR on the image
        extracted_text = pytesseract.image_to_string(image_bgr)
        return extracted_text

    # Extract text from the uploaded image
    extracted_text = extract_text_from_image(img_bgr)

    # Display the extracted text in Streamlit
    st.subheader("Extracted Text:")
    st.write(extracted_text)
