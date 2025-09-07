import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (replace 'yolov8n.pt' with your trained weights if any)
model = YOLO("yolov8n.pt")  # Or your custom trained model, e.g., 'best.pt'

st.title("Car Detection App with YOLOv8")
st.write("Upload an image and the app will detect cars.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to numpy array (OpenCV format)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Run YOLOv8 inference
    results = model.predict(source=image_cv, imgsz=640)

    # Draw bounding boxes on the image
    annotated_image = results[0].plot()  # Annotated image as numpy array
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display annotated image
    st.image(annotated_image, caption="Detected Cars", use_column_width=True)

    # Show detected classes
    detected_classes = results[0].boxes.cls
    class_names = [model.names[int(cls)] for cls in detected_classes]

    if "car" in class_names:
        st.success("Car detected!")
    else:
        st.warning("No car detected.")
