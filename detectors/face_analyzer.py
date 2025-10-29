from typing import List, Dict, Any

import numpy as np


class FaceAnalyzer:
	"""Detect faces and analyze age and emotion within person regions.

	Lazily loads MTCNN and DeepFace to avoid heavy imports during GUI startup.
	"""

	def __init__(self) -> None:
		self._mtcnn = None
		self._deepface = None

	def _ensure_mtcnn(self):
		if self._mtcnn is None:
			try:
				from mtcnn import MTCNN  # local import to delay TensorFlow init
			except Exception as exc:
				raise RuntimeError("MTCNN is not available. Please ensure 'mtcnn' is installed.") from exc
			self._mtcnn = MTCNN()

	def _ensure_deepface(self):
		if self._deepface is None:
			try:
				from deepface import DeepFace  # local import to delay TensorFlow init
			except Exception as exc:
				raise RuntimeError("DeepFace is not available. Please ensure 'deepface' is installed.") from exc
			self._deepface = DeepFace

	def _analyze_face_age_emotion(self, face_bgr: np.ndarray) -> Dict[str, Any]:
		self._ensure_deepface()
		analysis = self._deepface.analyze(
			img_path=face_bgr,
			actions=["age", "emotion"],
			enforce_detection=False,
			detector_backend="skip",
			prog_bar=False,
		)
		# DeepFace may return a list if multiple faces; with 'skip' we expect a single dict
		if isinstance(analysis, list) and analysis:
			analysis = analysis[0]
		return {
			"age": int(analysis.get("age", -1)) if isinstance(analysis.get("age", None), (int, float)) else analysis.get("age", None),
			"dominant_emotion": analysis.get("dominant_emotion", None),
			"emotion_scores": analysis.get("emotion", {}),
		}

	def analyze(self, image_bgr: np.ndarray, person_detections: List[Dict], min_face_confidence: float = 0.80) -> List[Dict]:
		"""
		For each detected person, find faces and run age/emotion analysis.
		Returns a list of entries with keys:
		- person_bbox: [x1,y1,x2,y2]
		- person_score: float
		- faces: List[{ face_bbox, face_score, age, dominant_emotion, emotion_scores }]
		"""
		self._ensure_mtcnn()
		results: List[Dict] = []
		image_h, image_w = image_bgr.shape[:2]
		for det in person_detections:
			x1, y1, x2, y2 = det["bbox"]
			x1, y1 = max(0, x1), max(0, y1)
			x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)
			person_roi = image_bgr[y1:y2, x1:x2]
			faces_raw = self._mtcnn.detect_faces(person_roi)

			faces_info = []
			for f in faces_raw:
				confidence = float(f.get("confidence", 0.0))
				if confidence < min_face_confidence:
					continue
				x, y, w, h = f.get("box", [0, 0, 0, 0])
				fx1, fy1 = max(0, x), max(0, y)
				fx2, fy2 = max(0, x + w), max(0, y + h)
				# Clip to ROI
				fx1, fy1 = min(fx1, person_roi.shape[1] - 1), min(fy1, person_roi.shape[0] - 1)
				fx2, fy2 = min(fx2, person_roi.shape[1] - 1), min(fy2, person_roi.shape[0] - 1)
				face_crop = person_roi[fy1:fy2, fx1:fx2]
				if face_crop.size == 0:
					continue

				try:
					analysis = self._analyze_face_age_emotion(face_crop)
					faces_info.append({
						"face_bbox": [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2],
						"face_score": confidence,
						"age": analysis.get("age"),
						"dominant_emotion": analysis.get("dominant_emotion"),
						"emotion_scores": analysis.get("emotion_scores", {}),
					})
				except Exception:
					# Skip problematic faces but continue processing others
					continue

			results.append({
				"person_bbox": [x1, y1, x2, y2],
				"person_score": float(det.get("score", 0.0)),
				"faces": faces_info,
			})

		return results
