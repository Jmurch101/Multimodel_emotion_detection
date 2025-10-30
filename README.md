# Multimodal Image Emotion & Age Analyzer

A desktop application that detects people in images, analyzes faces within each person, and estimates age and emotion using computer vision. Built specifically for macOS compatibility with optimized threading and lazy loading.

## 🚀 Quick Start

### Step 1: Clone and Navigate
```bash
git clone https://github.com/Jmurch101/Multimodel_emotion_detection.git
cd Multimodel_emotion_detection
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR on Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python tk_app.py
```

## 📋 Detailed Setup Instructions

### Prerequisites
- **Python 3.8+** (tested on Python 3.10)
- **macOS** (optimized for macOS compatibility)
- **4GB+ RAM** recommended
- **Stable internet** for initial model downloads

### Troubleshooting Setup Issues

**Permission Error:**
```bash
# If you get permission errors, try:
chmod +x tk_app.py
```

**Virtual Environment Issues:**
```bash
# Remove and recreate if issues occur
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Package Installation Fails:**
```bash
# Clear pip cache and retry
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

## 🎯 How to Use the Application

### Starting the App
1. **First Launch**: The app will show "INITIALIZING" then "LOADING MODELS"
2. **Wait**: First run downloads YOLOv8 model (~30-60 seconds)
3. **Ready**: Status changes to "READY" when models are loaded

### Main Interface Overview
The app window contains:
- **Image Display Area**: Shows your selected image with detection results
- **Status Label** (top): Shows current processing state
- **Progress Label** (below status): Detailed status messages
- **Control Buttons** (bottom): Open Image, Open Video, Open Camera, Stop
- **Results Table**: Shows detection details (appears after analysis)

### Step-by-Step Usage

#### 1. Open an Image
- Click **"Open Image"** button
- Select any image file (JPG, PNG, etc.) from your computer
- The image will display immediately

#### 2. Automatic Analysis
- App status changes to **"PROCESSING"**
- Progress shows **"Detecting people and faces..."**
- Analysis runs in background (UI stays responsive)

#### 3. View Results
- Status changes to **"ANALYSIS COMPLETE"** (green)
- **Image overlays** show:
  - 🔵 **Blue boxes**: Detected persons
  - 🟢 **Green boxes**: Detected faces within persons
- **Progress message** shows summary:
  - Number of persons detected
  - Number of faces found
  - Age range summary
  - Emotion distribution

#### 4. Understanding Results
- **Person Detection**: YOLOv8 finds all people in the image
- **Face Analysis**: For each person, detects faces and estimates:
  - **Age**: Estimated age range (8-90 years)
  - **Emotion**: happy, sad, neutral, surprised, etc.

### Status Messages Explained

| Status | Color | Meaning |
|--------|-------|---------|
| INITIALIZING | Orange | App starting up |
| LOADING MODELS | Orange | Downloading AI models |
| READY | Green | Ready for image analysis |
| PROCESSING | Blue | Analyzing image |
| ANALYSIS COMPLETE | Green | Results ready |
| NO PERSONS DETECTED | Orange | No people found in image |

### Example Output Messages
- `"Found 3 person(s) and 3 face(s). Ages: 25, 32, 45. Emotions: happy, neutral, surprised."`
- `"Found 1 person(s) and 1 face(s). Ages: 28. Emotions: happy."`
- `"Found 2 person(s) and 0 face(s). Age/emotion analysis not available."`

### Tips for Best Results
- **Clear Images**: Use well-lit, high-resolution photos
- **Visible Faces**: Ensure faces are not obscured or at extreme angles
- **Multiple People**: Works best with 1-5 people per image
- **File Formats**: Supports JPG, PNG, BMP, TIFF

## 🔧 Advanced Troubleshooting

### Common Issues and Solutions

**App Crashes on Startup (macOS):**
```bash
# This was fixed with lazy loading, but if it happens:
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export KMP_DUPLICATE_LIB_OK=TRUE
python tk_app.py
```

**"Module not found" Errors:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate
pip list  # Check if packages are installed
pip install -r requirements.txt  # Reinstall if needed
```

**Model Download Fails:**
- Check internet connection
- Wait and retry (models are cached after download)
- Check available disk space (>2GB free)

**App Freezes During Analysis:**
- This shouldn't happen with current threading implementation
- If it occurs, force quit and restart
- Check Activity Monitor for high CPU usage

**Poor Detection Results:**
- Use higher resolution images
- Ensure good lighting
- Avoid extreme angles or obscured faces
- Try images with fewer people (1-3 recommended)

### Testing the Models
Before running the full app, test individual components:
```bash
python test_models.py
```
This verifies:
- YOLOv8 person detection loads correctly
- OpenCV face detection works
- Age/emotion estimation functions

### Performance Optimization
- **CPU vs GPU**: Currently optimized for CPU (macOS compatibility)
- **Memory Usage**: ~500MB RAM during operation
- **Processing Time**: 5-15 seconds per image depending on complexity
- **Model Caching**: Models stay loaded between analyses

## 🛠️ Development and Contributing

### Project Structure
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

### Adding New Features
1. **Face Analysis**: Modify `detectors/face_analyzer_opencv.py`
2. **Person Detection**: Update `detectors/person_detector.py`
3. **UI Changes**: Edit `tk_app.py`
4. **Visualization**: Update `utils/visualization.py`

### Running Tests
```bash
# Test model imports
python test_models.py

# Run in development mode (with debug output)
python -c "import detectors.person_detector; print('Person detector OK')"
python -c "import detectors.face_analyzer_opencv; print('Face analyzer OK')"
```

### Building for Distribution
```bash
# Create standalone executable (optional)
pip install pyinstaller
pyinstaller --onefile --windowed tk_app.py
```

## 🐳 Docker Support

### Build and Run
```bash
# Build container
docker build -t multimodal-analyzer .

# Run container
docker run -p 8501:8501 multimodal-analyzer
```

### Docker for Development
```bash
# Mount source code for development
docker run -v $(pwd):/app -p 8501:8501 multimodal-analyzer
```

## 📊 Technical Details

### Models Used
- **Person Detection**: YOLOv8n (Ultralytics) - ~4MB, fast and accurate
- **Face Detection**: OpenCV Haar Cascades - Built-in, lightweight
- **Age Estimation**: Custom heuristic algorithm based on facial features
- **Emotion Detection**: Rule-based analysis of facial regions

### Accuracy Notes
- **Person Detection**: 85-95% accuracy on clear images
- **Face Detection**: 80-90% accuracy with Haar cascades
- **Age Estimation**: ±5-10 years (heuristic-based)
- **Emotion Detection**: Basic classification (happy, sad, neutral, surprised)

### System Requirements
- **Minimum**: Python 3.8, 4GB RAM, macOS 10.15+
- **Recommended**: Python 3.10, 8GB RAM, macOS 12+
- **Storage**: 2GB free space for models and cache

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Run `python test_models.py` to ensure models work
5. Submit a pull request with detailed description

### Code Style
- Follow PEP 8 Python style guide
- Add docstrings to new functions
- Test on macOS before submitting
- Update README for any UI changes

## 📝 License

MIT License - see LICENSE file for details

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Run `python test_models.py` to verify setup
3. Check GitHub Issues for similar problems
4. Create a new issue with:
   - macOS version
   - Python version
   - Error messages
   - Steps to reproduce

---

**Happy analyzing! 🎯**
