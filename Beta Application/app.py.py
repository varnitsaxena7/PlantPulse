import streamlit as st
import cv2
from keras.models import load_model
import google.generativeai as genai
from PIL import Image
import tensorflow as tf

genai.configure(api_key="AIzaSyBOisPhVp7vcjWXkcyU1KEQEiUvdhCiBIE")
if "finder" not in st.session_state:
    st.session_state.finder = ""
if "prediction_content" not in st.session_state:
    st.session_state.prediction_content = ""

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result

model = load_model('plant_disease.h5')

class_indices = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']

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
uploaded_image = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])
it = st.checkbox("Cure and Prevention Details")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_img = image.resize((150, 150))
    st.image(resized_img)

    if st.button('Predict'):
        prediction = predict_image_class(model, uploaded_image, class_indices)
        st.session_state.finder = prediction

        prediction_text = f"The name of this plant disease is {prediction}."
        hindi_prompt = f"[Hindi Name for Plant Disease {prediction} is]"
        hindi_name = gemini_pro_response(hindi_prompt)
        hindi_name_text = f"Hindi Name for this disease is {hindi_name}"

        st.session_state.prediction_content = f"{prediction_text}\n\n{hindi_name_text}"

        if it:
            with st.spinner('Thinking...'):
                user_prompt = f"[Prevention and Cure for Plant Disease {prediction}]"
                prevention_response = gemini_pro_response(user_prompt)
                st.session_state.prediction_content += f"\n\n{prevention_response}"

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
