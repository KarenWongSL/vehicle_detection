import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model
model = torch.load("my_model.pt", map_location="cpu")
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("Car Detection App")
st.write("Upload an image and the app will detect if it is a car.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Display result (assuming 1 = car, 0 = not car)
    if prediction == 1:
        st.success("This is a car!")
    else:
        st.error("This is not a car.")