import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import google.generativeai as genai
from googleapiclient.discovery import build

genai.configure(api_key="AIzaSyBOisPhVp7vcjWXkcyU1KEQEiUvdhCiBIE")

YOUTUBE_API_KEY = "AIzaSyBrSrLW5Xm6WAe6FZJ52pyEuu4rhCB96xU"

if "finder" not in st.session_state:
    st.session_state.finder = ""
if "prediction_content" not in st.session_state:
    st.session_state.prediction_content = ""

st.set_page_config(
    page_title="PlantPulse",
    page_icon="🌱",
    layout="centered",  
)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    return response.text

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def get_youtube_videos(query, api_key, max_results=5):
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_response = youtube.search().list(
        q=query,
        part='snippet',
        maxResults=max_results,
        type='video'
    ).execute()

    video_results = []
    for item in search_response['items']:
        video_results.append({
            'title': item['snippet']['title'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            'thumbnail': item['snippet']['thumbnails']['default']['url']
        })
    return video_results

st.title('Plant Disease Detector 🌱')

uploaded_image = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])
it = st.checkbox("Cure and Prevention Details")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_img = image.resize((150, 150))
    st.image(resized_img, caption="Uploaded Plant Leaf", use_column_width=True)

    if st.button('⚙ Predict'):
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

    st.subheader("Recommended YouTube Videos")
    videos = get_youtube_videos(st.session_state.finder, YOUTUBE_API_KEY)

    if videos:
        cols = st.columns(len(videos))
        for col, video in zip(cols, videos):
            with col:
                st.image(video['thumbnail'], use_column_width=True)
                st.markdown(f"**{video['title']}**")
                st.markdown(f"[Watch on YouTube]({video['url']})", unsafe_allow_html=True)
    else:
        st.write("No videos found for this disease.")

    st.subheader("Ask Your Doubts About This Disease")
    user_query = st.text_input("❓ Enter your question:")

    if user_query:
        if st.button("❗ Get Answer"):
            with st.spinner('⚙ Generating response...'):
                qa_prompt = f"[Answer the following question about the plant disease {st.session_state.finder}: {user_query}]"
                qa_response = gemini_pro_response(qa_prompt)
                if qa_response:
                    st.markdown(f"**Answer:** {qa_response}")
                else:
                    st.error("No response received. Please try again.")
