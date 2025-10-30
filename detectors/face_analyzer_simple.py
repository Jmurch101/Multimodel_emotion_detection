from typing import List, Dict, Any

import numpy as np


class SimpleFaceAnalyzer:
	"""Simple face analyzer using MTCNN for detection only (no age/emotion for now)"""

	def __init__(self) -> None:
		self.face_detector = None

	def _ensure_detector(self):
		if self.face_detector is None:
			try:
				from mtcnn import MTCNN
				self.face_detector = MTCNN()
			except Exception as exc:
				raise RuntimeError("MTCNN is not available. Please ensure 'mtcnn' is installed.") from exc

	def analyze(self, image_bgr: np.ndarray, person_detections: List[Dict], min_face_confidence: float = 0.80) -> List[Dict]:
		"""
		For each detected person, find faces using MTCNN only (no age/emotion analysis).
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
			faces_raw = self.face_detector.detect_faces(person_roi)

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

				faces_info.append({
					"face_bbox": [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2],
					"face_score": confidence,
					"age": None,  # Not available in simple version
					"dominant_emotion": None,  # Not available in simple version
					"emotion_scores": {},  # Not available in simple version
				})

			results.append({
				"person_bbox": [x1, y1, x2, y2],
				"person_score": float(det.get("score", 0.0)),
				"faces": faces_info,
			})

		return results
