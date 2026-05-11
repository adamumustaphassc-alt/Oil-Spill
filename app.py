import streamlit as st
from PIL import Image

from model.model import shallownet
from services.inference import load_model, predict
from utils.preprocessing import preprocess

# UI SETUP
st.set_page_config(page_title="Oil spill", layout="centered")

st.title("💎 Oil spill")
st.write("Upload satelite image.")

# LOAD MODEL
@st.cache_resource
def get_model():
    try:
        return load_model()
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = get_model()

# CLASS LABELS(As per the model's training)
# idx_to_class = {
#     0: "antelope_duiker",
#     1: "bird",
#     2: "blank",
#     3: "civet_genet",
#     4: "hog",
#     5: "leopard",
#     6: "monkey_prosimian",
#     7: "rodent"
# }

# FILE UPLOAD
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.warning("Model not loaded.")
    else:
        tensor = preprocess(image)
        result, confidence = predict(model, tensor)
        # result, confidence = predict(model, tensor, idx_to_class)

        st.success(f"Prediction: **{result}**")
        st.write(f"Confidence: {confidence:.2%}")