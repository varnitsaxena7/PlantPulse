import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import google.generativeai as genai
import os
import json
from PIL import Image
import tensorflow as tf

genai.configure(api_key="AIzaSyBOisPhVp7vcjWXkcyU1KEQEiUvdhCiBIE")

if "finder" not in st.session_state:
    st.session_state.finder = ""
if "prediction_content" not in st.session_state:
    st.session_state.prediction_content = ""

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease.h5"
model = tf.keras.models.load_model(model_path)

class_indices = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result


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
        predicted_class = class_indices[np.argmax(Y_pred)]
        st.session_state.finder = predicted_class
        st.markdown(f"<p class='result'>This is {predicted_class.split('-')[0]} leaf with {predicted_class.split('-')[1]}</p>", unsafe_allow_html=True)
        prediction_text = f"The name of this plant disease is {predicted_class}."
        hindi_prompt = f"[Hindi Name for Plant Disease {predicted_class} is]"
        hindi_name = gemini_pro_response(hindi_prompt)
        hindi_name_text = f"Hindi Name for this disease is {hindi_name}"

        st.session_state.prediction_content = f"{prediction_text}\n\n{hindi_name_text}"
        if it:
            with st.spinner('Thinking...'):
                user_prompt = f"[Prevention and Cure for Plant Disease {predicted_class}]"
                response = gemini_pro_response(user_prompt)
                st.session_state.prediction_content += f"\n\n{response}"




if st.session_state.prediction_content:
    st.markdown("### Prediction Details")
    st.markdown(st.session_state.prediction_content)

    st.subheader("Ask your doubts about this disease")
    user_query = st.text_input("Enter your question:")

    if user_query:
        if st.button("Get Answer"):
            with st.spinner('Generating response...'):
                qa_prompt = f"[Answer the following question about the plant disease {st.session_state.finder}: {user_query}]"
                qa_response = gemini_pro_response(qa_prompt)
                if qa_response:
                    st.markdown(f"**Answer:** {qa_response}")
                else:
                    st.error("No response received. Please try again.")
