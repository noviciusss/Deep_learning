# app.py
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@st.cache_resource
def load_model():
    # Use your saved path, e.g., "cifar10_best.keras" or a SavedModel directory
    return tf.keras.models.load_model("CNN/Cifar_10.h5")

model = load_model()

st.title("CIFAR-10 Image Classifier")
img_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg","jpeg","png"])
if img_file:
    img = Image.open(img_file).convert("RGB").resize((32, 32))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)

    preds = model.predict(arr)
    class_id = int(np.argmax(preds, axis=-1))
    conf = float(np.max(preds, axis=-1))

    st.image(img, caption=f"Pred: {CLASS_NAMES[class_id]} ({conf*100:.1f}%)", width=256)
