import os
# Constrain threads; apply before heavy imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Use a local Ultralytics settings directory to avoid system AppDirs
os.environ.setdefault("ULTRALYTICS_SETTINGS_DIR", os.path.join(os.getcwd(), "ultralytics_settings"))

from typing import Any, Dict

import numpy as np

# Lazy OpenCV config
def _ensure_cv2():
	import cv2 as _cv2
	try:
		_cv2.setNumThreads(1)
		try:
			_cv2.ocl.setUseOpenCL(False)
		except Exception:
			pass
	except Exception:
		pass
	return _cv2


def run_worker(conn) -> None:
	"""Run loop: receive dict commands, send dict responses.
	Commands:
	- {cmd: 'analyze', image: np.ndarray (BGR)} -> {ok: True, persons_with_faces: list}
	- {cmd: 'ping'} -> {ok: True}
	- {cmd: 'shutdown'} -> {ok: True} and exit
	"""
	cv2 = _ensure_cv2()
	person_detector = None
	face_analyzer = None

	# Ensure local settings dir exists
	try:
		os.makedirs(os.environ["ULTRALYTICS_SETTINGS_DIR"], exist_ok=True)
	except Exception:
		pass

	from detectors import PersonDetector  # import inside worker
	from detectors.face_analyzer import FaceAnalyzer

	while True:
		msg: Dict[str, Any] = conn.recv()
		cmd = msg.get("cmd")
		if cmd == "shutdown":
			conn.send({"ok": True})
			break
		if cmd == "ping":
			conn.send({"ok": True})
			continue
		if cmd == "analyze":
			try:
				if person_detector is None:
					person_detector = PersonDetector()
				if face_analyzer is None:
					face_analyzer = FaceAnalyzer()
				image_bgr: np.ndarray = msg["image"]
				persons = person_detector.detect(image_bgr)
				persons_with_faces = face_analyzer.analyze(image_bgr, persons)
				conn.send({"ok": True, "persons_with_faces": persons_with_faces})
			except Exception as exc:
				conn.send({"ok": False, "error": str(exc)})
			continue
		# Unknown command
		conn.send({"ok": False, "error": f"unknown cmd: {cmd}"})
