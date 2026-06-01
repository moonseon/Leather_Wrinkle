import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.title("Seat Wrinkle AI-lite")

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Input")

    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    gradient = np.sqrt(sobelx**2 + sobely**2)

    edge_density = np.mean(edges > 0)
    gradient_mean = np.mean(gradient)

    st.write("Edge density:", edge_density)
    st.write("Gradient:", gradient_mean)

    if edge_density > 0.08 or gradient_mean > 20:
        st.error("NG - Wrinkle detected")
    else:
        st.success("OK - Smooth")

    st.image(edges, caption="Edge Map")
