import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -----------------------------
# Load models
# -----------------------------
# YOLOv8 (pretrained or your custom trained weights)
yolo_model = YOLO("yolov8n.pt")

# Haar Cascade (your uploaded file path)
cascade_path = "HaarClassifier.xml"
haar_model = cv2.CascadeClassifier(cascade_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Car Detection App")
st.write("Choose between YOLOv8 or Haar Cascade for vehicle detection.")

model_choice = st.radio("Select Model", ("YOLOv8", "Haar Cascade"))

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # -----------------------------
    # YOLOv8 Inference
    # -----------------------------
    if model_choice == "YOLOv8":
        results = yolo_model.predict(image, conf=0.25)

        for r in results:
            annotated_img = r.plot()
            st.image(annotated_img, caption="YOLOv8 Detection", use_column_width=True)

            # Check if 'car' detected
            names = yolo_model.names
            detected_classes = [names[int(cls)] for cls in r.boxes.cls]
            if "car" in detected_classes:
                st.success("✅ A car was detected (YOLOv8)!")
            else:
                st.error("❌ No car detected (YOLOv8).")

    # -----------------------------
    # Haar Cascade Inference
    # -----------------------------
    elif model_choice == "Haar Cascade":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        cars = haar_model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in cars:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB),
                 caption=f"Haar Cascade Detection ({len(cars)} cars found)",
                 use_column_width=True)

        if len(cars) > 0:
            st.success("✅ A car was detected (Haar Cascade)!")
        else:
            st.error("❌ No car detected (Haar Cascade).")
