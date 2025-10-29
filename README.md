# Multimodal Image Emotion & Age Analyzer

Detect people in an image/video, then detect faces within each person and estimate age and emotion for each face.

## Features
- Person detection via YOLOv8
- Face detection via MTCNN within each person box
- Age and emotion analysis via DeepFace
- Two GUIs:
  - PyQt6 desktop app (`qt_app.py`) for images, videos, and camera
  - PySide6 desktop app (`qt_app_pyside.py`) alternative (use if PyQt6 has issues)
  - Streamlit web app (`app.py`) for quick local web UI (optional)

## Setup
```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
# Full stack (web + desktop)
pip install -r requirements.txt
# Or minimal desktop-only stack (no streamlit/pandas/pyarrow)
# pip install -r requirements-qt.txt
```

Notes:
- First run will download pretrained model weights (YOLO, DeepFace backends).
- CPU is fine by default.

## Run the PyQt desktop app
```bash
python qt_app.py
```

## Run the PySide desktop app (alternative)
```bash
# If needed: pip install PySide6
python qt_app_pyside.py
```

## Run the Streamlit app (optional)
```bash
streamlit run app.py
```

## Troubleshooting macOS GUI startup
- If the PyQt6 app aborts early, try:
  - Ensure no Conda base is active: `conda deactivate`
  - Use the minimal desktop requirements: `pip install -r requirements-qt.txt`
  - Try the PySide6 app: `python qt_app_pyside.py`

## Project structure
```
.
├── app.py                   # Streamlit app (optional)
├── qt_app.py                # PyQt6 desktop app (image/video/camera)
├── qt_app_pyside.py         # PySide6 desktop app alternative
├── detectors/
│   ├── __init__.py
│   ├── face_analyzer.py
│   └── person_detector.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── requirements.txt         # full stack
├── requirements-qt.txt      # minimal desktop-only stack
├── README.md
├── .gitignore
├── .dockerignore
├── Dockerfile
└── .github/workflows/ci.yml
```

## GitHub
```bash
git add .
git commit -m "Add PySide6 GUI alternative and minimal desktop requirements"
git push
```

## License
MIT
