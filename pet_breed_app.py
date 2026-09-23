import json
import pathlib

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_DIR = pathlib.Path(__file__).parent / "saved_model"
MODEL_PATH = MODEL_DIR / "pet_breed_mobilenetv2.keras"
CONFIG_PATH = MODEL_DIR / "preprocessing_config.json"


@st.cache_resource
def load_model_and_config():
    if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
        return None, None
    model = keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, config


def preprocess_image(pil_image: Image.Image, img_size):
    img = pil_image.convert("RGB").resize(tuple(img_size))
    arr = np.asarray(img, dtype=np.float32)
    arr = preprocess_input(
        arr
    )  # matches training preprocessing (mobilenet_v2 -> [-1, 1])
    return np.expand_dims(arr, axis=0)


def format_breed_name(raw_name: str) -> str:
    return raw_name.replace("_", " ").title()


def main():
    st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐾")
    st.title("🐾 Pet Breed Classifier")
    st.caption(
        "Upload a photo of a cat or dog and this model (fine-tuned MobileNetV2, "
        "trained on the Oxford-IIIT Pet Dataset) will predict the breed."
    )

    model, config = load_model_and_config()
    if model is None:
        st.error(
            "No saved model found. Run the training notebook (`oxford_pets_cnn.ipynb`) "
            "through Section 11 first, so that `saved_model/pet_breed_mobilenetv2.keras` "
            "and `saved_model/preprocessing_config.json` exist next to this app."
        )
        st.stop()

    idx_to_breed = {int(k): v for k, v in config["idx_to_breed"].items()}
    img_size = config["img_size"]

    uploaded_file = st.file_uploader("Upload a pet photo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Predicting..."):
            batch = preprocess_image(pil_image, img_size)
            probs = model.predict(batch, verbose=0)[0]

        top_k = 3
        top_indices = np.argsort(probs)[::-1][:top_k]

        with col2:
            best_idx = top_indices[0]
            best_breed = format_breed_name(idx_to_breed[best_idx])
            st.metric(
                "Predicted breed",
                best_breed,
                f"{probs[best_idx] * 100:.1f}% confidence",
            )

            st.write("**Top 3 predictions:**")
            for idx in top_indices:
                breed = format_breed_name(idx_to_breed[idx])
                st.progress(
                    float(probs[idx]), text=f"{breed} — {probs[idx] * 100:.1f}%"
                )

    st.divider()
    st.caption(
        "Model: MobileNetV2 (transfer learning + fine-tuning) · "
        "Dataset: Oxford-IIIT Pet Dataset, 37 breeds · "
        "This is a coursework prototype, not a veterinary tool."
    )


if __name__ == "__main__":
    main()
