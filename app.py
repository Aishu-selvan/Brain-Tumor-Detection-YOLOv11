import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

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

    st.subheader("Supported Tumor Types")
    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• Pituitary")

    st.markdown("---")

    st.subheader("About")
    st.write(
        """
        This application detects and localizes
        brain tumors from MRI scans using a
        YOLOv11 object detection model.
        """
    )

# ---------------- TITLE ----------------
st.title("🧠 Brain Tumor Detection using YOLOv11")

st.write(
    "Upload an MRI brain scan to detect and localize Glioma, Meningioma, or Pituitary tumors."
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
            conf=0.70
        )

    result = results[0]

    plotted = result.plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

    # ---------------- SHOW IMAGES ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original MRI")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        st.image(plotted, use_container_width=True)

    st.markdown("---")

    # ---------------- DETECTION SUMMARY ----------------

    if len(result.boxes) == 0:

        st.success("✅ No confident tumor detection")

        st.info(
            """
The model did not detect **Glioma, Meningioma, or Pituitary tumor**
with sufficient confidence.

**Important:**

This model was trained only on three tumor classes.

It was **not trained on normal MRI scans**, therefore the absence of
a detection should **not** be interpreted as a medical diagnosis.
"""
        )

    else:

        box = result.boxes[0]

        confidence = float(box.conf[0])

        if confidence < 0.70:

            st.warning("⚠️ Low-confidence prediction")

            st.info(
                """
The model produced a low-confidence prediction.

To reduce false positives, predictions below **70% confidence**
are not considered reliable.
"""
            )

        else:

            cls = int(box.cls[0])
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

            st.write("### Confidence Score")

            st.progress(confidence)

            if confidence >= 0.85:
                st.success("🟢 High Confidence Detection")
            else:
                st.warning("🟡 Moderate Confidence Detection")

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
## 📊 Model Information

- **Model:** YOLOv11
- **Framework:** Ultralytics
- **Input Size:** 640 × 640
- **Supported Classes:** Glioma, Meningioma, Pituitary
- **Deployment:** Streamlit
"""
)

st.warning(
"""
⚠️ **Disclaimer**

This application is developed for **educational and portfolio purposes only**.

The model detects only **Glioma**, **Meningioma**, and **Pituitary** tumors.

It **was not trained on normal MRI scans**, so the absence of a detection
does **not** confirm that an MRI is normal. The application should not be
used as a substitute for professional medical diagnosis.
"""
)

st.markdown("---")
st.caption("Developed by **Aiswarya T**")