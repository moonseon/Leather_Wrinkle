import streamlit as st
from PIL import Image
import sys

st.title("Seat Wrinkle AI (Ultra Stable)")
st.write("Python version:", sys.version)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_model()

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Input")

    if model:
        results = model.predict(img)
        st.image(results[0].plot(), caption="Detection")

        if len(results[0].boxes) > 0:
            st.error("NG - Wrinkle detected")
        else:
            st.success("OK - No defect")
    else:
        st.warning("Model not loaded")
