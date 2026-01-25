import streamlit as st
import requests
from PIL import Image
import io

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    layout="centered"
)

st.title("🧠 Brain Tumor Classification from MRI")
st.write(
    "Upload a brain MRI image to classify the tumor type. "
    "This system acts as a decision-support tool."
)

# -------------------------
# Upload image
# -------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# FastAPI endpoint
API_URL = "https://brain-tumor-classification-2911.onrender.com/predict"

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing MRI..."):
            try:
                # Send image to FastAPI
                files = {
                    "file": uploaded_file.getvalue()
                }

                response = requests.post(
                    API_URL,
                    files={"file": uploaded_file.getvalue()}
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction Successful ✅")
                    st.write(f"**Prediction:** {result['prediction']}")
                    st.write(f"**Confidence:** {result['confidence']}")

                else:
                    st.error("API error occurred.")
                    st.write(response.text)

            except Exception as e:
                st.error("Failed to connect to the API.")
                st.write(str(e))

# -------------------------
# Disclaimer
# -------------------------
st.markdown("---")
st.warning(
    "⚠️ **Disclaimer:** This application is intended for educational and "
    "research purposes only. It is NOT a medical diagnostic tool."
)
