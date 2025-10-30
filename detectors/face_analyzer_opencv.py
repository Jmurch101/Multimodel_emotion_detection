from typing import List, Dict, Any

import numpy as np


class OpenCVFaceAnalyzer:
	"""Face analyzer using OpenCV Haar cascades (no TensorFlow dependency)"""

	def __init__(self) -> None:
		self.face_cascade = None

	def _ensure_detector(self):
		if self.face_cascade is None:
			try:
				import cv2
				# Try to load Haar cascade classifier
				# This should work without TensorFlow
				cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
				self.face_cascade = cv2.CascadeClassifier(cascade_path)
				if self.face_cascade.empty():
					raise RuntimeError("Could not load Haar cascade classifier")
			except Exception as exc:
				raise RuntimeError("OpenCV Haar cascade not available. Please ensure OpenCV is installed.") from exc

	def analyze(self, image_bgr: np.ndarray, person_detections: List[Dict], min_face_confidence: float = 0.5) -> List[Dict]:
		"""
		For each detected person, find faces using OpenCV Haar cascades.
		Returns a list of entries with keys:
		- person_bbox: [x1,y1,x2,y2]
		- person_score: float
		- faces: List[{ face_bbox, face_score }]
		"""
		self._ensure_detector()
		results: List[Dict] = []
		image_h, image_w = image_bgr.shape[:2]

		for det in person_detections:
			x1, y1, x2, y2 = det["bbox"]
			x1, y1 = max(0, x1), max(0, y1)
			x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)
			person_roi = image_bgr[y1:y2, x1:x2]

			# Convert to grayscale for Haar cascades
			import cv2
			gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

			# Detect faces in ROI
			faces_rects = self.face_cascade.detectMultiScale(
				gray_roi,
				scaleFactor=1.1,
				minNeighbors=3,
				minSize=(30, 30),
				maxSize=(person_roi.shape[1], person_roi.shape[0])
			)

			faces_info = []
			for (fx, fy, fw, fh) in faces_rects:
				# Haar cascades don't give confidence scores, so we use a fixed score
				confidence = 0.8  # Fixed confidence for Haar cascades

				if confidence < min_face_confidence:
					continue

				fx1, fy1 = max(0, fx), max(0, fy)
				fx2, fy2 = max(0, fx + fw), max(0, fy + fh)
				# Clip to ROI
				fx1, fy1 = min(fx1, person_roi.shape[1] - 1), min(fy1, person_roi.shape[0] - 1)
				fx2, fy2 = min(fx2, person_roi.shape[1] - 1), min(fy2, person_roi.shape[0] - 1)

				faces_info.append({
					"face_bbox": [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2],
					"face_score": confidence,
					"age": None,  # Not available in OpenCV version
					"dominant_emotion": None,  # Not available in OpenCV version
					"emotion_scores": {},  # Not available in OpenCV version
				})

			results.append({
				"person_bbox": [x1, y1, x2, y2],
				"person_score": float(det.get("score", 0.0)),
				"faces": faces_info,
			})

		return results
