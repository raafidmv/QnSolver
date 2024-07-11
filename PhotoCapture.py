import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_cropper import st_cropper
import base64
import openai
import os

# Set the OpenAI API key
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Helper functions
def opencv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def pil_to_opencv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

st.title("Camera Capture with Cropping")

# Capture an image using the camera
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert the image to OpenCV format
    image = np.array(Image.open(img_file_buffer))

    # Display the captured image
    st.image(image, caption="Captured Image", use_column_width=True)

    # Provide cropping functionality
    st.write("Crop the image")
    cropped_image = st_cropper(opencv_to_pil(image))

    # Convert cropped image back to OpenCV format
    cropped_image = pil_to_opencv(cropped_image)

    # Display the cropped image
    st.image(cropped_image, caption="Cropped Image", use_column_width=True)

    # Save the cropped image as aw.png
    save_button = st.button("Save Cropped Image")
    if save_button:
        cropped_image_path = "aw.png"
        cv2.imwrite(cropped_image_path, cropped_image)
        st.success(f"Cropped image saved as {cropped_image_path}")

        # Encode the image
        
