import streamlit as st
from predict import predict
import os

st.set_page_config(page_title="Plant Disease Detection")

st.title("🌿 Plant Disease Detection")

uploaded = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    os.makedirs("uploads",exist_ok=True)

    filepath="uploads/test.jpg"

    with open(filepath,"wb") as f:
        f.write(uploaded.getbuffer())

    st.image(uploaded,width=500)

    disease,confidence=predict(filepath)

    st.success(f"Prediction : {disease}")

    st.info(f"Confidence : {confidence*100:.2f}%")