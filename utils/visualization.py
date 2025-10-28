from typing import List, Dict

import cv2
import numpy as np


def _draw_label(image: np.ndarray, text: str, x: int, y: int, color=(0, 0, 0), bg=(255, 255, 255)) -> None:
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.5
	thickness = 1
	(text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
	pad = 4
	top_left = (x, max(0, y - text_h - baseline - pad * 2))
	bottom_right = (x + text_w + pad * 2, y)
	cv2.rectangle(image, top_left, bottom_right, bg, thickness=cv2.FILLED)
	cv2.putText(image, text, (x + pad, y - pad), font, scale, color, thickness, cv2.LINE_AA)


def draw_annotations(image_bgr: np.ndarray, persons_with_faces: List[Dict]) -> np.ndarray:
	"""Draw person and face boxes with labels on a copy of the image and return it."""
	annotated = image_bgr.copy()
	person_color = (255, 128, 0)  # Blue-ish
	face_color = (0, 200, 0)      # Green

	for idx, entry in enumerate(persons_with_faces):
		x1, y1, x2, y2 = entry["person_bbox"]
		person_score = entry.get("person_score", 0.0)
		cv2.rectangle(annotated, (x1, y1), (x2, y2), person_color, 2)
		label = f"Person {idx+1}: {person_score*100:.1f}%"
		_draw_label(annotated, label, x1, y1)

		for f_idx, face in enumerate(entry.get("faces", [])):
			fx1, fy1, fx2, fy2 = face["face_bbox"]
			cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), face_color, 2)
			age = face.get("age", None)
			emo = face.get("dominant_emotion", None)
			face_score = face.get("face_score", 0.0)
			parts = []
			if age is not None and isinstance(age, (int, float)) and age >= 0:
				parts.append(f"Age: {int(age)}")
			if emo:
				parts.append(str(emo).title())
			parts.append(f"{face_score*100:.0f}%")
			face_label = " | ".join(parts) if parts else f"Face {f_idx+1}"
			_draw_label(annotated, face_label, fx1, fy1)

	return annotated
