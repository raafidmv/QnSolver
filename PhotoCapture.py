import streamlit as st
import cv2
from PIL import Image
import numpy as np
from streamlit_cropper import st_cropper
import base64
from openai import OpenAI
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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
    cropped_image = st_cropper(opencv_to_pil(image), aspect_ratio=None)

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
        base64_image = encode_image(cropped_image_path)

        # Create the OpenAI client
        client = OpenAI()

        # Generate the response
        response = client.chat_completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in Physics, Mathematics, and Chemistry. Answer the following question by determining the subject first. "
                        "If the question is related to Physics, respond as Phys, an expert in Physics. If the question is related to Mathematics, respond as Math, an expert in Mathematics. "
                        "If the question is related to Biology, respond as Bio, an expert in Biology. If the question is related to Chemistry, respond as Chem, an expert in Chemistry. "
                        "Ensure that explanations are thorough, detailed, and easy to understand. If the question is not related to these subjects, respond with I don't know. "
                        "Consider the following: - Use clear and concise language. - Break down complex concepts into simpler steps. - Provide examples where possible. "
                        "- Use relevant formulas, equations, and scientific principles. - If the question is ambiguous, ask for clarification."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Solve the question?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=0.0,
        )

        # Get the response content
        response_content = response.choices[0].message.content

        # Display the response content as Markdown for LaTeX rendering
        st.markdown(response_content)
