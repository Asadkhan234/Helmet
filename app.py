import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
import warnings

warnings.filterwarnings('ignore')


# ================= HELMET DETECTOR CLASS =================

class HelmetDetector:
    def __init__(self, model_path='best.pt', conf_threshold=0.25, iou_threshold=0.45):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.model_path = model_path
        self.confidence_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        try:
            st.info(f"Loading helmet detection model: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)
            st.success(f"Model loaded successfully on {self.device}")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    def predict(self, image):
        if self.model is None:
            st.warning("Model not loaded.")
            return None, image

        # Convert PIL → OpenCV
        if isinstance(image, Image.Image):
            image = np.array(image)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            start = time.time()
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device
            )
            end = time.time()

            st.info(f"Inference time: {round((end-start)*1000, 2)} ms")

            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            return results, annotated
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return None, image

    def extract_detections(self, results):
        detections = []
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    detections.append({
                        "bbox": box.xyxy[0].cpu().numpy().tolist(),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "class_name": results[0].names[int(box.cls[0])]
                    })
        return detections


# ================= STREAMLIT UI =================

st.set_page_config(page_title="Helmet Detection System", layout="wide")
st.title("🪖 Helmet Detection System")

option = st.radio("Select input type", ["Image", "Video"])

detector = HelmetDetector("best.pt", conf_threshold=0.25)

# ================= IMAGE MODE =================

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        results, annotated_image = detector.predict(image)

        if annotated_image is not None:
            st.image(annotated_image, caption="Helmet Detection Result", use_column_width=True)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(tmp.name, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            st.download_button("Download Result", tmp.name, "helmet_detected.png")

        if results:
            st.subheader("Detections")
            st.json(detector.extract_detections(results))


# ================= VIDEO MODE =================

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(
            tmp_out.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        stframe = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            results, annotated = detector.predict(frame)
            out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            stframe.image(annotated, channels="RGB")
            progress.progress((i + 1) / total_frames)

        cap.release()
        out.release()

        st.success("Video processing completed!")
        st.video(tmp_out.name)
        st.download_button("Download Video", tmp_out.name, "helmet_detected_video.mp4")
