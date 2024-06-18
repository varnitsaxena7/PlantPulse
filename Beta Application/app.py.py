import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import google.generativeai as genai
genai.configure(api_key="AIzaSyBOisPhVp7vcjWXkcyU1KEQEiUvdhCiBIE")
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result
model = load_model('plant_disease.h5')
CLASS_NAMES = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
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

plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
it=st.checkbox("Prevention and Cure Details")
if st.button('Predict', key='predict_button'):

    if plant_image is not None:
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR", caption="Uploaded Image", width=300)

        resized_image = cv2.resize(opencv_image, (256, 256))

        input_image = np.expand_dims(resized_image, axis=0)

        Y_pred = model.predict(input_image)
        predicted_class = CLASS_NAMES[np.argmax(Y_pred)]

        st.markdown(f"<p class='result'>This is {predicted_class.split('-')[0]} leaf with {predicted_class.split('-')[1]}</p>", unsafe_allow_html=True)

        if it:
            with st.spinner('Thinking...'):
                user_prompt = f"[Prevention and Cure for Plant Disease {predicted_class}]"
                response = gemini_pro_response(user_prompt)
                st.markdown(response)
