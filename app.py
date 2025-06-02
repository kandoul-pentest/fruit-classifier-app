import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("model_nadia_brahim.keras")

# Define class names
class_names = [
    'apple fruit',
    'banana fruit',
    'cherry fruit',
    'chickoo fruit',
    'grapes fruit',
    'kiwi fruit',
    'mango fruit',
    'orange fruit',
    'strawberry fruit'
]

st.set_page_config(page_title="Fruit Classifier 🍍", layout="wide")
st.title("🍉 Fruit Image Classifier")
st.write("Upload an image of a fruit and the model will tell you what it is.")

uploaded_file = st.file_uploader("Upload a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    
    # Preprocess
    resized_img = img.resize((224, 224))
    img_array = image.img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.markdown(f"""
        <div style="padding-top: 100px;">
            <h1 style="font-size: 48px; color: #4CAF50; font-weight: bold;">
                {predicted_class}
            </h1>
            <p style="font-size: 24px; color: #333;">
                Confidence: <strong>{confidence:.2f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
