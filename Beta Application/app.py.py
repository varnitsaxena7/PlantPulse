import streamlit as st
import cv2
from keras.models import load_model
import google.generativeai as genai
from PIL import Image
import tensorflow as tf

# Configure the Google Generative AI with your API key
genai.configure(api_key="AIzaSyBOisPhVp7vcjWXkcyU1KEQEiUvdhCiBIE")

# Function to get response from Gemini Pro model
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result

# Load the pre-trained model
model = load_model('plant_disease.h5')

# Define class names
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

# Streamlit app UI setup
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload {
        text-align: center;
        margin-bottom: 20px;
    }
    .predict-button {
        display: block;
        margin: 20px auto;
        padding: 10px 20px;
        font-size: 18px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .image-container {
        text-align: center;
        margin-top: 20px;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .details {
        margin-top: 20px;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("Upload an image of the plant leaf", unsafe_allow_html=True)

# File uploader for plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
it = st.checkbox("Prevention and Cure Details")

# Predict button
if st.button('Predict', key='predict_button'):

    if plant_image is not None:
        image = Image.open(plant_image)
        opencv_image = cv2.cvtColor(tf.keras.preprocessing.image.img_to_array(image), cv2.COLOR_RGB2BGR)
        
        # Normalize image data to [0.0, 1.0]
        opencv_image = opencv_image / 255.0
        
        st.image(opencv_image, channels="BGR", caption="Uploaded Image", width=300)

        resized_image = cv2.resize(opencv_image, (256, 256))
        input_image = resized_image.reshape((1, 256, 256, 3))

        Y_pred = model.predict(input_image)
        predicted_class = CLASS_NAMES[tf.argmax(Y_pred[0])]

        st.markdown(f"<p class='result'>This is {predicted_class.split('-')[0]} leaf with {predicted_class.split('-')[1]}</p>", unsafe_allow_html=True)

        if it:
            with st.spinner('Thinking...'):
                user_prompt = f"[Prevention and Cure for Plant Disease {predicted_class}]"
                response = gemini_pro_response(user_prompt)
                st.markdown(response)
