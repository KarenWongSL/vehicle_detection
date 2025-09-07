import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model (you can replace with your custom trained model path)
model = YOLO("yolov8n.pt")  # yolov8n.pt = small pretrained model

st.title("Car Detection App (YOLO)")
st.write("Upload an image and the app will detect if it contains a car.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO inference
    results = model.predict(image, conf=0.25)  # confidence threshold

    # Display YOLO annotated results
    for r in results:
        annotated_img = r.plot()  # draw boxes
        st.image(annotated_img, caption="Detection Result", use_column_width=True)

        # Check if 'car' is detected
        names = model.names  # class names
        detected_classes = [names[int(cls)] for cls in r.boxes.cls]

        if "car" in detected_classes:
            st.success("A car was detected in the image!")
        else:
            st.error("No car detected in the image.")
