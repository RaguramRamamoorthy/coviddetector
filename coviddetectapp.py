import streamlit as st
from PIL import Image, ImageOps
from imageclassification import predict_image_class


st.title("COVID 19 Detector App")
st.text("Upload a Chext X-Ray for diagnosis")

uploaded_file = st.file_uploader("Choose a Chest X-Ray....",type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded X-ray image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    probabilities, predicted_class_index, predicted_class_name = predict_image_class(image)
    st.write(predicted_class_name)

st.text("If you need Chest-Xray images for testing the App")
st.text("download a zip folder from here...")

with open("images.zip", "rb") as fp:
    btn = st.download_button(
        label="Download ZIP",
        data=fp,
        file_name="images.zip",
        mime="application/zip"
    )