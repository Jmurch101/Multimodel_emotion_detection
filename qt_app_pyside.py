import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from typing import List, Dict

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
	QApplication,
	QFileDialog,
	QHBoxLayout,
	QLabel,
	QMainWindow,
	QMessageBox,
	QPushButton,
	QTableWidget,
	QTableWidgetItem,
	QVBoxLayout,
	QWidget,
)

from detectors import PersonDetector
from detectors.face_analyzer import FaceAnalyzer
from utils import draw_annotations


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


def bgr_to_qimage(image_bgr: object) -> QImage:
	cv2 = _ensure_cv2()
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	h, w, ch = image_rgb.shape
	bytes_per_line = ch * w
	qimg = QImage(
		image_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
	)
	return qimg.copy()


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Multimodal Emotion & Age Analyzer (PySide6)")

		self.person_detector = None
		self.face_analyzer = None

		self.video_cap = None  # type: ignore
		self.timer = QTimer(self)
		self.timer.timeout.connect(self._on_timer)
		self.frame_count = 0
		self.infer_every_n = 5
		self.last_persons_with_faces: List[Dict] = []

		self.image_label = QLabel()
		self.image_label.setAlignment(Qt.AlignCenter)
		self.image_label.setMinimumSize(640, 360)

		self.btn_open_image = QPushButton("Open Image")
		self.btn_open_video = QPushButton("Open Video")
		self.btn_open_camera = QPushButton("Open Camera")
		self.btn_stop = QPushButton("Stop")

		self.btn_open_image.clicked.connect(self.open_image)
		self.btn_open_video.clicked.connect(self.open_video)
		self.btn_open_camera.clicked.connect(self.open_camera)
		self.btn_stop.clicked.connect(self.stop_stream)

		controls = QHBoxLayout()
		controls.addWidget(self.btn_open_image)
		controls.addWidget(self.btn_open_video)
		controls.addWidget(self.btn_open_camera)
		controls.addWidget(self.btn_stop)
		controls.addStretch()

		self.table = QTableWidget(0, 6)
		self.table.setHorizontalHeaderLabels([
			"person_id", "person_conf", "face_id", "face_conf", "age", "emotion"
		])
		self.table.horizontalHeader().setStretchLastSection(True)

		layout = QVBoxLayout()
		layout.addWidget(self.image_label)
		layout.addLayout(controls)
		layout.addWidget(self.table)

		container = QWidget()
		container.setLayout(layout)
		self.setCentralWidget(container)

	def _ensure_detectors(self) -> bool:
		if self.person_detector is None:
			try:
				self.person_detector = PersonDetector()
			except Exception as exc:
				QMessageBox.critical(self, "Error", f"Failed to init person detector: {exc}")
				return False
		if self.face_analyzer is None:
			try:
				self.face_analyzer = FaceAnalyzer()
			except Exception as exc:
				QMessageBox.critical(self, "Error", f"Failed to init face analyzer: {exc}")
				return False
		return True

	def open_image(self) -> None:
		cv2 = _ensure_cv2()
		path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
		if not path:
			return
		image_bgr = cv2.imread(path)
		if image_bgr is None:
			return
		self._process_and_display(image_bgr)

	def open_video(self) -> None:
		cv2 = _ensure_cv2()
		path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
		if not path:
			return
		self._start_capture(_ensure_cv2().VideoCapture(path))

	def open_camera(self) -> None:
		cv2 = _ensure_cv2()
		self._start_capture(cv2.VideoCapture(0))

	def stop_stream(self) -> None:
		if self.timer.isActive():
			self.timer.stop()
		if self.video_cap is not None:
			self.video_cap.release()
			self.video_cap = None

	def _start_capture(self, cap) -> None:
		self.stop_stream()
		self.video_cap = cap
		if not self.video_cap.isOpened():
			self.video_cap = None
			return
		self.frame_count = 0
		self.timer.start(33)

	def _on_timer(self) -> None:
		if self.video_cap is None:
			self.stop_stream()
			return
		ok, frame = self.video_cap.read()
		if not ok or frame is None:
			self.stop_stream()
			return
		self.frame_count += 1
		if self.frame_count % self.infer_every_n == 0:
			self._process_and_display(frame)
		else:
			to_show = frame
			if self.last_persons_with_faces:
				to_show = draw_annotations(frame, self.last_persons_with_faces)
			self._display_only(to_show)

	def _process_and_display(self, image_bgr: object) -> None:
		if not self._ensure_detectors():
			return
		persons = self.person_detector.detect(image_bgr)
		persons_with_faces = self.face_analyzer.analyze(image_bgr, persons)
		self.last_persons_with_faces = persons_with_faces
		annotated = draw_annotations(image_bgr, persons_with_faces)
		self._display_only(annotated)
		self._update_table(persons_with_faces)

	def _display_only(self, image_bgr: object) -> None:
		qimg = bgr_to_qimage(image_bgr)
		self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
			self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
		))

	def _update_table(self, persons_with_faces: List[Dict]) -> None:
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

		self.table.setRowCount(len(rows))
		for r, row in enumerate(rows):
			self.table.setItem(r, 0, QTableWidgetItem(str(row.get("person_id", ""))))
			self.table.setItem(r, 1, QTableWidgetItem(str(row.get("person_conf", ""))))
			self.table.setItem(r, 2, QTableWidgetItem(str(row.get("face_id", ""))))
			self.table.setItem(r, 3, QTableWidgetItem(str(row.get("face_conf", ""))))
			self.table.setItem(r, 4, QTableWidgetItem(str(row.get("age", ""))))
			self.table.setItem(r, 5, QTableWidgetItem(str(row.get("emotion", ""))))


if __name__ == "__main__":
	app = QApplication(sys.argv)
	w = MainWindow()
	w.resize(1000, 800)
	w.show()
	sys.exit(app.exec())
