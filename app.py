import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = YOLO("best.pt")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🧠 Brain Tumor AI")

    st.markdown("---")

    st.subheader("Model")
    st.success("YOLOv11")

    st.subheader("Supported Tumors")
    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• Pituitary")

    st.markdown("---")

    st.subheader("About")
    st.write(
        """
        This application detects and localizes
        brain tumors from MRI scans using
        a trained YOLOv11 model.
        """
    )

# ---------------- TITLE ----------------
st.title("🧠 Brain Tumor Detection using YOLOv11")

st.write(
    "Upload an MRI scan to detect and localize brain tumors using Deep Learning."
)

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    uploaded_file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("🔍 Detecting Tumor..."):

        results = model.predict(
            temp_path,
            conf=0.25
        )

    result = results[0]

    plotted = result.plot()

    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # ---------------- SIDE BY SIDE ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(plotted, use_container_width=True)

    # ---------------- PREDICTION CARD ----------------

    st.markdown("---")

    if len(result.boxes) > 0:

        box = result.boxes[0]

        cls = int(box.cls[0])

        confidence = float(box.conf[0])

        tumor = model.names[cls]

        st.subheader("📋 Detection Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Tumor Type",
            tumor.capitalize()
        )

        c2.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        c3.metric(
            "Objects Detected",
            len(result.boxes)
        )

        st.write("### Confidence")

        st.progress(confidence)

        if confidence > 0.90:
            st.success("🟢 High Confidence Detection")

        elif confidence > 0.75:
            st.warning("🟡 Medium Confidence Detection")

        else:
            st.error("🔴 Low Confidence Detection")

    else:

        st.success("✅ No Tumor Detected")

    # ---------------- DOWNLOAD ----------------

    output_path = "prediction_result.jpg"

    cv2.imwrite(
        output_path,
        cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
    )

    with open(output_path, "rb") as file:

        st.download_button(
            label="📥 Download Prediction",
            data=file,
            file_name="prediction_result.jpg",
            mime="image/jpeg"
        )

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    """
### 📊 Model Information

- **Model:** YOLOv11
- **Input Size:** 640 × 640
- **Framework:** Ultralytics YOLO
- **Classes:** Glioma, Meningioma, Pituitary
- **Deployment:** Streamlit

---

Developed by **Aiswarya T**
"""
)