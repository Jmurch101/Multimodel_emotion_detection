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

# Minimal imports only
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
from typing import List, Dict, Optional
import threading


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
	from PIL import Image
	cv2 = _ensure_cv2()
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	return Image.fromarray(image_rgb)


def _ensure_piltk():
	from PIL import ImageTk
	return ImageTk


class InferenceEngine:
	def __init__(self) -> None:
		self.person_detector = None
		self.face_analyzer = None
		self._is_loading = False
		self._load_error = None

	def load_models(self) -> None:
		if self.person_detector is not None and self.face_analyzer is not None:
			return
		if self._is_loading:
			return

		self._is_loading = True
		self._load_error = None
		try:
			# Lazy import ML modules only when needed
			if self.person_detector is None:
				from detectors import PersonDetector
				self.person_detector = PersonDetector()
			if self.face_analyzer is None:
				# Use OpenCV face analyzer (no TensorFlow) for macOS compatibility
				from detectors.face_analyzer_opencv import OpenCVFaceAnalyzer
				self.face_analyzer = OpenCVFaceAnalyzer()
		except Exception as e:
			self._load_error = str(e)
		finally:
			self._is_loading = False

	def is_loading(self) -> bool:
		return self._is_loading

	def get_load_error(self) -> Optional[str]:
		return self._load_error

	def analyze(self, image_bgr) -> Optional[List[Dict]]:
		if self.person_detector is None or self.face_analyzer is None:
			return None
		try:
			persons = self.person_detector.detect(image_bgr)
			persons_with_faces = self.face_analyzer.analyze(image_bgr, persons)
			return persons_with_faces
		except Exception:
			return None

	def draw_annotations(self, image_bgr, persons_with_faces):
		# Lazy import draw_annotations function
		from utils import draw_annotations
		return draw_annotations(image_bgr, persons_with_faces)


class MainApp:
	def __init__(self, root):
		self.root = root
		self.root.title("Multimodal Emotion & Age Analyzer (Tkinter)")

		self.infer = InferenceEngine()
		self.video_cap = None
		self.timer_id = None
		self.infer_thread = None
		self.frame_count = 0
		self.infer_every_n = 5
		self.last_persons_with_faces: List[Dict] = []
		self._pending_image = None

		# UI
		self.image_label = tk.Label(root)
		self.image_label.pack(pady=10)

		self.status_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
		self.status_label.pack(pady=5)

		self.progress_label = tk.Label(root, text="", fg="blue")
		self.progress_label.pack()

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

		# Load models in background
		self.status_label.config(text="INITIALIZING", fg="orange")
		self.progress_label.config(text="Loading AI models (may take 30-60 seconds first time)...")
		self.root.after(100, self._load_models_async)

	def _load_models_async(self) -> None:
		if self.infer.is_loading():
			self.root.after(500, self._load_models_async)
			return

		if self.infer.person_detector is None or self.infer.face_analyzer is None:
			# Start loading in background thread
			load_thread = threading.Thread(target=self.infer.load_models, daemon=True)
			load_thread.start()
			self.root.after(500, self._load_models_async)
		else:
			# Check for load error
			error = self.infer.get_load_error()
			if error:
				self.status_label.config(text="MODEL LOAD FAILED", fg="red")
				self.progress_label.config(text=f"Error: {error}")
				messagebox.showerror("Model Load Error", f"Failed to load models: {error}")
			else:
				self.status_label.config(text="READY", fg="green")
				self.progress_label.config(text="Models loaded successfully. Ready to process images.")

	def open_image(self) -> None:
		if self.infer.person_detector is None or self.infer.face_analyzer is None:
			messagebox.showerror("Error", "Models are still loading. Please wait.")
			return

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
		if self.infer.person_detector is None or self.infer.face_analyzer is None:
			messagebox.showerror("Error", "Models are still loading. Please wait.")
			return

		cv2 = _ensure_cv2()
		path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv")])
		if not path:
			return
		self._start_capture(cv2.VideoCapture(path))

	def open_camera(self) -> None:
		if self.infer.person_detector is None or self.infer.face_analyzer is None:
			messagebox.showerror("Error", "Models are still loading. Please wait.")
			return

		cv2 = _ensure_cv2()
		self._start_capture(cv2.VideoCapture(0))

	def stop_stream(self) -> None:
		if self.timer_id:
			self.root.after_cancel(self.timer_id)
			self.timer_id = None
		if self.infer_thread and self.infer_thread.is_alive():
			# Wait for inference to finish
			self.infer_thread.join(timeout=1)
		self.infer_thread = None
		if self.video_cap is not None:
			self.video_cap.release()
			self.video_cap = None
		self.status_label.config(text="READY", fg="green")
		self.progress_label.config(text="Ready to process images.")

	def on_close(self) -> None:
		if self.infer_thread and self.infer_thread.is_alive():
			self.infer_thread.join(timeout=1)
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
		if (self.frame_count % self.infer_every_n == 0) and (self.infer_thread is None or not self.infer_thread.is_alive()):
			self._process_and_display(frame)
		else:
			to_show = frame if not self.last_persons_with_faces else self.infer.draw_annotations(frame, self.last_persons_with_faces)
			self._display_only(to_show)
		self.timer_id = self.root.after(33, self._on_timer)

	def _inference_done(self, result: Optional[List[Dict]]) -> None:
		self.infer_thread = None
		if result is None:
			self.status_label.config(text="Inference failed")
			messagebox.showerror("Error", "Inference failed")
			return

		self.last_persons_with_faces = result
		if self._pending_image is not None:
			annotated = self.infer.draw_annotations(self._pending_image, result)
			self._display_only(annotated)
			self._pending_image = None
		self._update_table(result)

		# Count detections
		total_persons = len(result)
		total_faces = sum(len(entry.get("faces", [])) for entry in result)

		if total_persons == 0:
			self.status_label.config(text="NO PERSONS DETECTED", fg="orange")
			self.progress_label.config(text="No people found in the image. Try a different image with visible people.")
		else:
			self.status_label.config(text="ANALYSIS COMPLETE", fg="green")
			if total_faces > 0:
				# Show age/emotion summary
				ages = []
				emotions = []
				for entry in result:
					for face in entry.get("faces", []):
						if face.get("age"):
							ages.append(face["age"])
						if face.get("dominant_emotion"):
							emotions.append(face["dominant_emotion"])

				age_summary = f"Ages: {', '.join(map(str, ages))}" if ages else "Ages: N/A"
				emotion_summary = f"Emotions: {', '.join(emotions)}" if emotions else "Emotions: N/A"

				self.progress_label.config(text=f"Found {total_persons} person(s) and {total_faces} face(s). {age_summary}. {emotion_summary}.")
			else:
				self.progress_label.config(text=f"Found {total_persons} person(s) and {total_faces} face(s). Age/emotion analysis not available.")

	def _process_and_display(self, image_bgr) -> None:
		self._display_only(image_bgr)
		self._pending_image = image_bgr
		self.status_label.config(text="PROCESSING", fg="blue")
		self.progress_label.config(text="Detecting people and faces...")

		if self.infer_thread and self.infer_thread.is_alive():
			return  # Already processing

		def inference_task():
			result = self.infer.analyze(image_bgr)
			self.root.after(0, lambda: self._inference_done(result))

		self.infer_thread = threading.Thread(target=inference_task, daemon=True)
		self.infer_thread.start()

	def _display_only(self, image_bgr) -> None:
		pil_img = bgr_to_pil(image_bgr)
		# Resize to fit label
		label_width, label_height = 640, 360
		pil_img.thumbnail((label_width, label_height))
		ImageTk = _ensure_piltk()
		tk_img = ImageTk.PhotoImage(pil_img)
		self.image_label.config(image=tk_img)
		self.image_label.image = tk_img  # keep reference

	def _update_table(self, persons_with_faces: List[Dict]) -> None:
		for item in self.table.get_children():
			self.table.delete(item)
		for p_idx, entry in enumerate(persons_with_faces):
			for f_idx, face in enumerate(entry.get("faces", [])):
				age_val = face.get("age")
				if age_val is None:
					age_display = "N/A"
				elif isinstance(age_val, (int, float)):
					age_display = str(int(age_val))
				else:
					age_display = str(age_val)

				emotion_val = face.get("dominant_emotion")
				emotion_display = emotion_val.title() if emotion_val else "N/A"

				values = (
					str(p_idx + 1),
					str(round(float(entry.get("person_score", 0.0)), 3)),
					str(f_idx + 1),
					str(round(float(face.get("face_score", 0.0)), 3)),
					age_display,
					emotion_display,
				)
				self.table.insert("", tk.END, values=values)


if __name__ == "__main__":
	root = tk.Tk()
	app = MainApp(root)
	root.mainloop()
