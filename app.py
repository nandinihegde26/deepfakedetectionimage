import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Custom CSS styles
def load_custom_styles():
    st.markdown("""
        <style>
            body {
                font-family: 'Arial', sans-serif;
            }
            .main {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #ffffff;
                padding: 2rem;
            }
            .stApp {
                background: transparent;
            }
            .header {
                text-align: center;
                padding: 2.5rem;
                background: rgba(22, 33, 62, 0.85);
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .header h1 {
                color: #60a5fa;
                font-size: 3rem;
                font-weight: bold;
                margin-bottom: 1rem;
            }
            .header p {
                color: #94a3b8;
                font-size: 1.25rem;
                max-width: 600px;
                margin: 0 auto;
            }
            
            .result-card {
                padding: 2rem;
                border-radius: 15px;
                margin: 2rem 0;
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid #3b82f6;
                animation: fadeIn 0.5s ease-out;
            }
            .footer {
                text-align: center;
                padding: 2rem 0;
                color: #94a3b8;
                border-top: 1px solid #1e293b;
                margin-top: 3rem;
            }
            .stButton>button {
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                color: white;
                border: none;
                padding: 0.75rem 2rem;
                border-radius: 9999px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    """, unsafe_allow_html=True)

class DeepFakeDetector:
    def __init__(self, model_path='deepfake_detection_model.h5'):
        try:
            self.model = load_model(model_path)
        except:
            st.error("Model file not found. Using simulation mode.")
            self.model = None

    def preprocess_image(self, image):
        image = cv2.resize(image, (96, 96))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image

    def predict(self, image):
        if self.model is None:
            # Simulation mode
            import random
            return {
                'result': random.choice(["Real", "Fake"]),
                'confidence': random.uniform(0.7, 0.99)
            }
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        class_label = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][class_label])

        return {
            'result': "Fake" if class_label == 0 else "Real",
            'confidence': confidence
        }

def render_header():
    st.markdown("""
        <div class="header">
            <h1>DeepGuard</h1>
            <p>AI-Powered Deepfake Detection</p>
        </div>
    """, unsafe_allow_html=True)

def render_upload_section():
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your image here or click to browse",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    if uploaded_file:
        file_bytes = uploaded_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Invalid image format. Please upload a valid JPG, JPEG, or PNG file.")
            return None
        st.image(image, channels="BGR", use_column_width=True)
        return image
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def render_results(result):
    result_color = "#22c55e" if result['result'] == "Real" else "#ef4444"
    st.markdown(f"""
        <div class="result-card" style="color: {result_color};">
            <h2>{result['result']} Content Detected</h2>
            <p>Confidence Score: {result['confidence']:.1%}</p>
        </div>
    """, unsafe_allow_html=True)

def render_footer():
    st.markdown("""
        <div class="footer">
            <p>DeepFake Detection AI.</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    load_custom_styles()
    detector = DeepFakeDetector()
    render_header()
    image = render_upload_section()
    if image is not None:
        result = detector.predict(image)
        render_results(result)
    else:
        st.info("Please upload an image to start the analysis.")
    render_footer()

if __name__ == "__main__":
    st.set_page_config(
        page_title="DeepFake Detection AI",
        page_icon="🔍",
        layout="wide"
    )
    main()
