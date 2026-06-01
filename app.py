import streamlit as st
from PIL import Image

st.title("Seat Wrinkle AI (Stable Version)")

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

model = load_model()

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Input")

    results = model.predict(img)

    st.image(results[0].plot(), caption="Detection")

    if len(results[0].boxes) > 0:
        st.error("NG - Wrinkle detected")
    else:
        st.success("OK - No defect")
