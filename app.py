import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8n.pt")

st.title("Seat Wrinkle AI")
file = st.file_uploader("Upload", type=["jpg","png"])

if file:
    img = Image.open(file)
    st.image(img)
    results = model.predict(img)
    st.image(results[0].plot())
    if len(results[0].boxes)>0:
        st.error("NG")
    else:
        st.success("OK")
