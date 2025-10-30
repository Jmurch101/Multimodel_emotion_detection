# Multimodal Image Emotion & Age Analyzer

Detect people in images, then detect faces within each person and estimate age and emotion using computer vision techniques. Built specifically for macOS compatibility.

## Features
- **Person Detection**: YOLOv8 for robust person detection
- **Face Detection**: OpenCV Haar cascades for reliable face detection
- **Age Estimation**: Custom OpenCV-based heuristic algorithm
- **Emotion Detection**: Facial feature analysis for emotion classification
- **Desktop App**: Tkinter-based GUI optimized for macOS
- **Status Indicators**: Real-time progress updates during processing
- **Results Display**: Clear visualization of detections with age/emotion summaries

## Setup
```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes:**
- First run will download YOLOv8 model weights (~30-60 seconds)
- CPU mode is enabled for macOS compatibility
- Uses lazy loading to prevent startup crashes on macOS

## Usage
```bash
python tk_app.py
```

**App Features:**
- **Status Updates**: Shows "INITIALIZING", "LOADING MODELS", "READY", "PROCESSING", "ANALYSIS COMPLETE"
- **Image Upload**: Click "Open Image" to analyze photos
- **Real-time Results**: Person detection boxes, face analysis, age/emotion summaries
- **macOS Optimized**: Threading and lazy loading prevent crashes
- **First Run**: Model download takes 30-60 seconds

## Project Structure
```
.
├── tk_app.py                    # Main Tkinter desktop application
├── app.py                       # Streamlit web app (legacy)
├── test_models.py               # Model testing script
├── detectors/
│   ├── __init__.py
│   ├── person_detector.py       # YOLOv8 person detection
│   ├── face_analyzer_opencv.py  # OpenCV face/age/emotion analysis
│   ├── face_analyzer.py         # Legacy MTCNN/DeepFace analyzer
│   └── face_analyzer_simple.py  # Simplified analyzer
├── utils/
│   ├── __init__.py
│   └── visualization.py         # Image annotation utilities
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .gitignore
├── Dockerfile                   # Containerization
├── .dockerignore
└── .github/workflows/ci.yml     # CI/CD pipeline
```

## Troubleshooting (macOS)

**Mutex Lock Errors**: Fixed with CPU-only mode and lazy loading
**Slow First Run**: Model downloads are cached after first use
**App Not Responding**: Background processing keeps UI responsive

## Development
- **Testing**: Run `python test_models.py` to verify models
- **CI/CD**: GitHub Actions tests imports on Python 3.10
- **Docker**: `docker build -t multimodal-analyzer .`

## GitHub Repository
```bash
git clone https://github.com/Jmurch101/Multimodel_emotion_detection.git
cd Multimodel_emotion_detection
pip install -r requirements.txt
python tk_app.py
```

## License
MIT
