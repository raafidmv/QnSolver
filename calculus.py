import streamlit as st
import base64
# import openai
import os
from openai import OpenAI
# Set OpenAI API key
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
openai.api_key = st.secrets["openai"]["api_key"]

client = OpenAI()

# Function to encode image as a base64 string
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

# Streamlit UI
st.title("Math Assistant")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    base64_image = encode_image(uploaded_image)

    # Show the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Create the OpenAI client
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    MODEL = "gpt-4o"

    # Generate the response
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that responds in Markdown. "
                    "You are Math, an expert in Calculus. Provide detailed and accurate comprehensive answers for complex calculus questions, "
                    "covering limits, derivatives, integrals, differential equations, and multivariable calculus. "
                    "Include detailed explanations, mathematical derivations, and practical applications."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Print the content shown in the image, then Solve the question?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.0,
    )

    # Get the response content
    response_content = response.choices[0].message.content

    # Display thearesponse content as Markdown for LaTeX rendering
    st.markdown(response_content)

