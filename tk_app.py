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

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
from typing import List, Dict, Optional
from PIL import Image, ImageTk
import multiprocessing as mp

from detectors import PersonDetector
from detectors.face_analyzer import FaceAnalyzer
from utils import draw_annotations


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


def bgr_to_pil(image_bgr):
	"""Convert BGR numpy array to PIL Image."""
	cv2 = _ensure_cv2()
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	return Image.fromarray(image_rgb)


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

	def request_analyze(self, image_bgr) -> bool:
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


class MainApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Multimodal Emotion & Age Analyzer (Tkinter)")

		self.infer = InferenceProcess()
		self.video_cap = None
		self.timer_id = None
		self.infer_timer_id = None
		self.frame_count = 0
		self.infer_every_n = 5
		self.last_persons_with_faces: List[Dict] = []
		self._pending_image = None

		# UI
		self.image_label = tk.Label(root)
		self.image_label.pack(pady=10)

		self.status_label = tk.Label(root, text="")
		self.status_label.pack()

		controls = tk.Frame(root)
		controls.pack(pady=5)

		self.btn_open_image = tk.Button(controls, text="Open Image", command=self.open_image)
		self.btn_open_image.pack(side=tk.LEFT, padx=5)

		self.btn_open_video = tk.Button(controls, text="Open Video", command=self.open_video)
		self.btn_open_video.pack(side=tk.LEFT, padx=5)

		self.btn_open_camera = tk.Button(controls, text="Open Camera", command=self.open_camera)
		self.btn_open_camera.pack(side=tk.LEFT, padx=5)

		self.btn_stop = tk.Button(controls, text="Stop", command=self.stop_stream)
		self.btn_stop.pack(side=tk.LEFT, padx=5)

		# Table
		columns = ("person_id", "person_conf", "face_id", "face_conf", "age", "emotion")
		self.table = ttk.Treeview(root, columns=columns, show="headings")
		for col in columns:
			self.table.heading(col, text=col)
			self.table.column(col, width=100)
		self.table.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

		self.root.protocol("WM_DELETE_WINDOW", self.on_close)

	def open_image(self) -> None:
		cv2 = _ensure_cv2()
		path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
		if not path:
			return
		image_bgr = cv2.imread(path)
		if image_bgr is None:
			messagebox.showerror("Error", "Could not load image")
			return
		self._process_and_display(image_bgr)

	def open_video(self) -> None:
		cv2 = _ensure_cv2()
		path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv")])
		if not path:
			return
		self._start_capture(cv2.VideoCapture(path))

	def open_camera(self) -> None:
		cv2 = _ensure_cv2()
		self._start_capture(cv2.VideoCapture(0))

	def stop_stream(self) -> None:
		if self.timer_id:
			self.root.after_cancel(self.timer_id)
			self.timer_id = None
		if self.infer_timer_id:
			self.root.after_cancel(self.infer_timer_id)
			self.infer_timer_id = None
		if self.video_cap is not None:
			self.video_cap.release()
			self.video_cap = None
		self.status_label.config(text="")

	def on_close(self) -> None:
		self.infer.stop()
		self.root.destroy()

	def _start_capture(self, cap) -> None:
		self.stop_stream()
		self.video_cap = cap
		if not self.video_cap.isOpened():
			self.video_cap = None
			messagebox.showerror("Error", "Could not open video/camera")
			return
		self.frame_count = 0
		self._on_timer()

	def _on_timer(self) -> None:
		if self.video_cap is None:
			self.stop_stream()
			return
		ok, frame = self.video_cap.read()
		if not ok or frame is None:
			self.stop_stream()
			return
		self.frame_count += 1
		if (self.frame_count % self.infer_every_n == 0) and (not self.infer.is_busy()):
			self._process_and_display(frame)
		else:
			to_show = frame if not self.last_persons_with_faces else draw_annotations(frame, self.last_persons_with_faces)
			self._display_only(to_show)
		self.timer_id = self.root.after(33, self._on_timer)

	def _on_infer_timer(self) -> None:
		resp = self.infer.poll_result()
		if resp is None:
			self.infer_timer_id = self.root.after(50, self._on_infer_timer)
			return
		self.infer_timer_id = None
		if not resp.get("ok"):
			self.status_label.config(text="Inference failed")
			messagebox.showerror("Error", f"Inference failed: {resp.get('error')}")
			return
		persons_with_faces = resp.get("persons_with_faces", [])
		self.last_persons_with_faces = persons_with_faces
		if self._pending_image is not None:
			annotated = draw_annotations(self._pending_image, persons_with_faces)
			self._display_only(annotated)
			self._pending_image = None
		self._update_table(persons_with_faces)
		self.status_label.config(text="")

	def _process_and_display(self, image_bgr) -> None:
		self._display_only(image_bgr)
		self._pending_image = image_bgr
		self.status_label.config(text="Analyzing...")
		if self.infer.request_analyze(image_bgr):
			if self.infer_timer_id is None:
				self._on_infer_timer()
		else:
			# Already busy
			pass

	def _display_only(self, image_bgr) -> None:
		pil_img = bgr_to_pil(image_bgr)
		# Resize to fit label
		label_width, label_height = 640, 360
		pil_img.thumbnail((label_width, label_height))
		tk_img = ImageTk.PhotoImage(pil_img)
		self.image_label.config(image=tk_img)
		self.image_label.image = tk_img  # keep reference

	def _update_table(self, persons_with_faces: List[Dict]) -> None:
		for item in self.table.get_children():
			self.table.delete(item)
		for p_idx, entry in enumerate(persons_with_faces):
			for f_idx, face in enumerate(entry.get("faces", [])):
				values = (
					str(p_idx + 1),
					str(round(float(entry.get("person_score", 0.0)), 3)),
					str(f_idx + 1),
					str(round(float(face.get("face_score", 0.0)), 3)),
					str(int(face.get("age", -1)) if isinstance(face.get("age", None), (int, float)) else face.get("age", None)),
					str(face.get("dominant_emotion", None)),
				)
				self.table.insert("", tk.END, values=values)


if __name__ == "__main__":
	# Ensure spawn context for macOS
	try:
		mp.set_start_method("spawn", force=True)
	except Exception:
		pass
	root = tk.Tk()
	app = MainApp(root)
	root.mainloop()
