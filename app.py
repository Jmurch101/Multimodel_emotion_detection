import io
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from detectors import PersonDetector
from detectors.face_analyzer import FaceAnalyzer
from utils import draw_annotations


st.set_page_config(page_title="Multimodal Image Emotion & Age Analyzer", layout="wide")
st.title("Multimodal Image Emotion & Age Analyzer")
st.write("Upload an image to detect people, then estimate age and emotion for each detected face.")

with st.sidebar:
	st.header("Settings")
	person_conf = st.slider("Person confidence threshold", 0.1, 0.9, 0.35, 0.05)
	person_iou = st.slider("Person NMS IoU threshold", 0.2, 0.9, 0.45, 0.05)
	face_conf = st.slider("Face confidence threshold (MTCNN)", 0.3, 0.99, 0.85, 0.01)

	st.markdown("""
	**Notes**
	- First run downloads model weights (YOLO, DeepFace). This can take a bit.
	- CPU is used by default; if you have a GPU, Ultralytics/DeepFace may use it.
	""")

# Lazy singletons in session state
if "person_detector" not in st.session_state:
	st.session_state.person_detector = PersonDetector()
if "face_analyzer" not in st.session_state:
	st.session_state.face_analyzer = FaceAnalyzer()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"]) 

if uploaded is not None:
	# Decode image bytes to BGR
	file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
	image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	if image_bgr is None:
		st.error("Could not read the uploaded image. Please try another file.")
		st.stop()

	col1, col2 = st.columns([3, 2])
	with col1:
		st.subheader("Input")
		st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)

	with st.spinner("Detecting people..."):
		persons = st.session_state.person_detector.detect(
			image_bgr,
			conf_threshold=person_conf,
			iou_threshold=person_iou,
		)

	with st.spinner("Detecting faces and analyzing age/emotion..."):
		persons_with_faces = st.session_state.face_analyzer.analyze(
			image_bgr, persons, min_face_confidence=face_conf
		)

	annotated = draw_annotations(image_bgr, persons_with_faces)

	with col1:
		st.subheader("Results")
		st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated", use_column_width=True)

	# Build a flat table of results
	rows: List[Dict] = []
	for p_idx, entry in enumerate(persons_with_faces):
		for f_idx, face in enumerate(entry.get("faces", [])):
			rows.append({
				"person_id": p_idx + 1,
				"person_conf": round(float(entry.get("person_score", 0.0)), 3),
				"face_id": f_idx + 1,
				"face_conf": round(float(face.get("face_score", 0.0)), 3),
				"age": int(face.get("age", -1)) if isinstance(face.get("age", None), (int, float)) else face.get("age", None),
				"emotion": face.get("dominant_emotion", None),
			})

	with col2:
		st.subheader("Detections")
		if rows:
			df = pd.DataFrame(rows)
			st.dataframe(df, use_container_width=True)
			st.caption(f"Persons: {len(persons_with_faces)} | Faces: {sum(len(e['faces']) for e in persons_with_faces)}")
		else:
			st.info("No faces found within detected persons.")
else:
	st.info("Upload an image to begin.")
