import os
# Constrain threads and fork behavior early (before heavy imports)
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
os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")

# Robustly set Qt plugin/framework paths for PyQt6 on macOS before importing Qt
try:
	import importlib.util as _ilu
	from pathlib import Path as _Path
	spec = _ilu.find_spec("PyQt6")
	if spec and spec.submodule_search_locations:
		_base = _Path(list(spec.submodule_search_locations)[0])
		_plugins_root = _base / "Qt6" / "plugins"
		_platforms = _plugins_root / "platforms"
		_frameworks = _base / "Qt6" / "lib"
		if _plugins_root.exists():
			os.environ.setdefault("QT_PLUGIN_PATH", str(_plugins_root))
		if _platforms.exists():
			os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(_platforms))
		if _frameworks.exists():
			# These help the platform plugin find Qt frameworks at runtime
			os.environ.setdefault("DYLD_FRAMEWORK_PATH", str(_frameworks))
			os.environ.setdefault("DYLD_LIBRARY_PATH", str(_frameworks))
except Exception:
	pass

import sys
from typing import List, Dict, Optional

import multiprocessing as mp

from PyQt6.QtCore import Qt, QTimer, QCoreApplication
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
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

from utils import draw_annotations

# Also add Qt library paths programmatically
try:
	# Add both plugins root and platforms to the Qt library search paths
	if 'QT_PLUGIN_PATH' in os.environ:
		QCoreApplication.addLibraryPath(os.environ['QT_PLUGIN_PATH'])
	if 'QT_QPA_PLATFORM_PLUGIN_PATH' in os.environ:
		QCoreApplication.addLibraryPath(os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'])
except Exception:
	pass


def _ensure_cv2():
	# Lazy import OpenCV and configure it to reduce threading issues
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
	return qimg.copy()  # ensure ownership


class InferenceProcess:
	def __init__(self) -> None:
		self._proc: Optional[mp.Process] = None
		self._parent_conn = None
		self._ctx = mp.get_context("spawn")
		self._pending = False

	def start(self) -> None:
		if self._proc is not None:
			return
		parent_conn, child_conn = self._ctx.Pipe()
		from inference_worker import run_worker
		self._proc = self._ctx.Process(target=run_worker, args=(child_conn,), daemon=True)
		self._proc.start()
		self._parent_conn = parent_conn
		self._parent_conn.send({"cmd": "ping"})
		self._parent_conn.recv()

	def is_busy(self) -> bool:
		return bool(self._pending)

	def request_analyze(self, image_bgr: object) -> bool:
		if self._proc is None or self._parent_conn is None:
			self.start()
		if self._pending:
			return False
		self._parent_conn.send({"cmd": "analyze", "image": image_bgr})
		self._pending = True
		return True

	def poll_result(self) -> Optional[Dict]:
		if self._proc is None or self._parent_conn is None:
			return None
		if self._parent_conn.poll():
			resp = self._parent_conn.recv()
			self._pending = False
			return resp
		return None

	def stop(self) -> None:
		try:
			if self._parent_conn is not None:
				self._parent_conn.send({"cmd": "shutdown"})
				self._parent_conn.recv()
			if self._proc is not None and self._proc.is_alive():
				self._proc.join(timeout=1)
		except Exception:
			pass
		finally:
			self._proc = None
			self._parent_conn = None
			self._pending = False


class MainWindow(QMainWindow):
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("Multimodal Emotion & Age Analyzer (PyQt)")

		self.infer = InferenceProcess()
		self.video_cap = None  # type: ignore
		self.timer = QTimer(self)
		self.timer.timeout.connect(self._on_timer)
		self.infer_timer = QTimer(self)
		self.infer_timer.timeout.connect(self._on_infer_timer)
		self.frame_count = 0
		self.infer_every_n = 5
		self.last_persons_with_faces: List[Dict] = []
		self._pending_image = None

		# UI
		self.image_label = QLabel()
		self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.image_label.setMinimumSize(640, 360)

		self.status_label = QLabel("")

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
		layout.addWidget(self.status_label)
		layout.addWidget(self.table)

		container = QWidget()
		container.setLayout(layout)
		self.setCentralWidget(container)

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
		self._start_capture(cv2.VideoCapture(path))

	def open_camera(self) -> None:
		cv2 = _ensure_cv2()
		self._start_capture(cv2.VideoCapture(0))

	def stop_stream(self) -> None:
		if self.timer.isActive():
			self.timer.stop()
		if self.infer_timer.isActive():
			self.infer_timer.stop()
		if self.video_cap is not None:
			self.video_cap.release()
			self.video_cap = None
		self.status_label.setText("")

	def closeEvent(self, event) -> None:  # type: ignore
		self.infer.stop()
		super().closeEvent(event)

	def _start_capture(self, cap) -> None:
		self.stop_stream()
		self.video_cap = cap
		if not self.video_cap.isOpened():
			self.video_cap = None
			return
		self.frame_count = 0
		self.timer.start(33)  # ~30 FPS

	def _on_timer(self) -> None:
		if self.video_cap is None:
			self.stop_stream()
			return
		ok, frame = self.video_cap.read()
		if not ok or frame is None:
			self.stop_stream()
			return
		self.frame_count += 1
		# Only request inference if not already running
		if (self.frame_count % self.infer_every_n == 0) and (not self.infer.is_busy()):
			self._process_and_display(frame)
		else:
			to_show = frame if not self.last_persons_with_faces else draw_annotations(frame, self.last_persons_with_faces)
			self._display_only(to_show)

	def _on_infer_timer(self) -> None:
		resp = self.infer.poll_result()
		if resp is None:
			return
		self.infer_timer.stop()
		if not resp.get("ok"):
			self.status_label.setText("Inference failed")
			QMessageBox.critical(self, "Error", f"Inference failed: {resp.get('error')}")
			return
		persons_with_faces = resp.get("persons_with_faces", [])
		self.last_persons_with_faces = persons_with_faces
		if self._pending_image is not None:
			annotated = draw_annotations(self._pending_image, persons_with_faces)
			self._display_only(annotated)
			self._pending_image = None
		self._update_table(persons_with_faces)
		self.status_label.setText("")

	def _process_and_display(self, image_bgr: object) -> None:
		# Show immediately while analysis runs
		self._display_only(image_bgr)
		self._pending_image = image_bgr
		self.status_label.setText("Analyzing...")
		if self.infer.request_analyze(image_bgr):
			if not self.infer_timer.isActive():
				self.infer_timer.start(50)
		else:
			# Already busy; will be picked up on next timer tick
			pass

	def _display_only(self, image_bgr: object) -> None:
		qimg = bgr_to_qimage(image_bgr)
		self.image_label.setPixmap(QPixmap.fromImage(qimg).scaled(
			self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
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
	# Ensure spawn context for macOS
	try:
		mp.set_start_method("spawn", force=True)
	except Exception:
		pass
	app = QApplication(sys.argv)
	w = MainWindow()
	w.resize(1000, 800)
	w.show()
	sys.exit(app.exec())
