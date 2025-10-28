from typing import List, Dict

import numpy as np

try:
	from ultralytics import YOLO
except Exception as exc:  # pragma: no cover
	YOLO = None  # type: ignore


class PersonDetector:
	"""Detect persons in an image using a pretrained YOLOv8 model."""

	_model = None

	def __init__(self, model_name: str = "yolov8n.pt") -> None:
		self.model_name = model_name

	def _ensure_model_loaded(self) -> None:
		if PersonDetector._model is None:
			if YOLO is None:
				raise RuntimeError(
					"Ultralytics YOLO is not available. Please ensure 'ultralytics' is installed."
				)
			PersonDetector._model = YOLO(self.model_name)

	def detect(self, image_bgr: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> List[Dict]:
		"""
		Runs person detection on a BGR image.

		Returns a list of dicts: { 'bbox': [x1, y1, x2, y2], 'score': float }
		"""
		self._ensure_model_loaded()
		results = PersonDetector._model.predict(
			image_bgr,
			verbose=False,
			conf=conf_threshold,
			iou=iou_threshold,
		)
		if not results:
			return []

		result = results[0]
		if result.boxes is None or len(result.boxes) == 0:
			return []

		# YOLO class 0 is 'person'
		boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N,4)
		classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
		scores = result.boxes.conf.cpu().numpy()  # (N,)

		detections: List[Dict] = []
		for box, cls_id, score in zip(boxes_xyxy, classes, scores):
			if int(cls_id) != 0:
				continue
			x1, y1, x2, y2 = [int(max(0, v)) for v in box.tolist()]
			detections.append({
				"bbox": [x1, y1, x2, y2],
				"score": float(score),
			})
		return detections
