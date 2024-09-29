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
import pytesseract
import cv2
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update if necessary

# Title
st.title("OCR Application")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Function to perform OCR
def extract_text_from_image(image):
    # Convert the uploaded file to an OpenCV image
    image = np.array(image)
    
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform OCR using pytesseract
    extracted_text = pytesseract.image_to_string(rgb_image)
    return extracted_text

# Check if the user has uploaded an image
if uploaded_image is not None:
    # Open the image using PIL
    image = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Extract and display text
    extracted_text = extract_text_from_image(image)
    st.subheader("Extracted Text:")
    st.write(extracted_text)

