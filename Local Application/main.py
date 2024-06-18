import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
genai.configure(api_key="AIzaSyBOisPhVp7vcjWXkcyU1KEQEiUvdhCiBIE")


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name
st.title('Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
it=st.checkbox("Cure and Prevention Details")
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_img = image.resize((150, 150))
    st.image(resized_img)
    if st.button('Predict'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}') 
            predictionn="The name of this plant disease is "+prediction+"."
            st.markdown(predictionn)
            use = f"[Hindi Name for Plant Disease {prediction} is]"
            res = gemini_pro_response(use)
            ress=f"Hindi Name for this disease is {res}"
            st.markdown(ress)
            if it:
                with st.spinner('Thinking...'):
                    user_prompt = f"[Prevention and Cure for Plant Disease {prediction}]"
                    response = gemini_pro_response(user_prompt)
                    st.markdown(response)