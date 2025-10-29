# Multimodal Image Emotion & Age Analyzer

Detect people in an image/video, then detect faces within each person and estimate age and emotion for each face.

## Features
- Person detection via YOLOv8
- Face detection via MTCNN within each person box
- Age and emotion analysis via DeepFace
- Tkinter desktop app (`tk_app.py`) for images, videos, and camera (macOS compatible)
- Streamlit web app (`app.py`) for quick local web UI (optional)

## Setup
```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Desktop app (recommended for macOS)
pip install -r requirements-qt.txt
# Or full stack (includes Streamlit web app)
# pip install -r requirements.txt
```

Notes:
- First run will download pretrained model weights (YOLO, DeepFace backends).
- CPU is fine by default.

## Run the Tkinter desktop app
```bash
python tk_app.py
```
- **macOS Compatible**: Uses lazy loading to avoid threading issues
- **First Run**: "Loading models..." (~30–90s to download YOLO/DeepFace models)
- **Buttons**: Open Image, Open Video, Open Camera, Stop
- **Features**: Real-time person detection, face analysis with age/emotion, results table
- **UI**: Image displays immediately, analysis runs in background, annotations overlay results

## Run the Streamlit app (optional)
```bash
streamlit run app.py
```

## Project structure
```
.
├── app.py                   # Streamlit web app (optional)
├── tk_app.py                # Tkinter desktop app (recommended)
├── qt_app.py                # PyQt6 desktop app (deprecated)
├── qt_app_pyside.py         # PySide6 desktop app (deprecated)
├── inference_worker.py      # Multiprocessing worker (legacy)
├── detectors/
│   ├── __init__.py
│   ├── face_analyzer.py
│   └── person_detector.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── requirements.txt         # full stack
├── requirements-qt.txt      # minimal desktop stack
├── README.md
├── .gitignore
├── .dockerignore
├── Dockerfile
└── .github/workflows/ci.yml
```

## GitHub
```bash
git add .
git commit -m "Final: Working Tkinter app with lazy loading for macOS"
git push
```

## License
MIT
