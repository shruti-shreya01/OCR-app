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
from PIL import Image
import pickle
import os
import numpy as np

# Set the path for the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
  # Update this path based on your local setup

# Function to extract text from an image
def extract_text_from_image(image):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use pytesseract to do OCR on the image for both Hindi and English
    extracted_text = pytesseract.image_to_string(rgb_image, lang='hin+eng')
    
    return extracted_text

# Load the model from the pickle file
def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    else:
        model = None
    return model

# Save the model to a pickle file
def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Streamlit app
def main():
    st.title("OCR for Hindi and English Text")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    # Path to save and load the model
    model_path = "ocr_model.pkl"
    
    # Load or initialize the model
    model = load_model(model_path)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Perform OCR using the model
        extracted_text = extract_text_from_image(image)
        
        st.write("Extracted Text:")
        st.write(extracted_text)
        
        # Save the model after processing
        save_model(model, model_path)

if __name__ == '__main__':
    main()

