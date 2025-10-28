# Multimodal Image Emotion & Age Analyzer

Detect people in an image, then detect faces within each person and estimate age and emotion for each face. Simple Streamlit GUI.

## Features
- Person detection via YOLOv8
- Face detection via MTCNN within each person box
- Age and emotion analysis via DeepFace
- Streamlit GUI for image upload and visualization

## Setup
```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- The first run will download pretrained model weights (YOLO, DeepFace backends).
- If you hit GPU/CUDA issues, use CPU by default; the code automatically works on CPU.

## Run the app
```bash
streamlit run app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

## Project structure
```
.
├── app.py
├── detectors/
│   ├── __init__.py
│   ├── face_analyzer.py
│   └── person_detector.py
├── utils/
│   ├── __init__.py
│   └── visualization.py
├── requirements.txt
├── README.md
└── .gitignore
```

## GitHub
Initialize and push to a new GitHub repository:
```bash
git init
git add .
git commit -m "Initial commit: multimodal person/face age+emotion analyzer"
# Create a new repo on GitHub first (via web UI), then:
git branch -M main
git remote add origin git@github.com:<your-username>/<your-repo>.git
git push -u origin main
```

## License
MIT
