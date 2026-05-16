# AI Detection Web App

A web application for detecting AI-generated or manipulated images and videos using state-of-the-art machine learning models.

## Features

- **Image Detection**: Upload images (JPG, PNG) to check for AI manipulation
- **Video Detection**: Upload videos (MP4, AVI) to analyze frames for deepfakes
- **Real-time Analysis**: Fast processing using Hugging Face transformers
- **Web Interface**: Simple drag-and-drop interface
- **REST API**: Backend API for programmatic access

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **AI Models**: Hugging Face Transformers
  - Image detection: `prithivMLmods/open-deepfake-detection`
  - Video detection: `tayyabimam/Deepfake`

## Installation

1. Clone or download this repository
2. Set up Python virtual environment (already configured)
3. Install dependencies (already installed)

## Running the Application

### Start the Backend

```bash
cd backend
"c:/Users/sujit/OneDrive/Documents/course work/AI DETECTION/.venv/Scripts/python.exe" app.py
```

The server will start on `http://localhost:8002` by default. To use a different port, set `PORT` before running the app.

### Open the Frontend

Open `frontend/index.html` in your web browser, or serve it with a local server.

## API Endpoints

- `POST /detect/image` - Detect AI manipulation in images
- `POST /detect/video` - Detect AI manipulation in videos
- `GET /` - API information

## Usage

1. Open the web interface
2. Drag and drop or select an image/video file
3. Click "Analyze File"
4. View the detection results

## Accuracy & Limitations

- **Accuracy**: 85-95% depending on the content
- **Limitations**:
  - May not detect highly sophisticated deepfakes
  - Performance varies with image/video quality
  - Requires good lighting and clear subjects for best results

## Models Used

All models are free and open-source from Hugging Face:
- [Open Deepfake Detection](https://huggingface.co/prithivMLmods/open-deepfake-detection)
- [Deepfake Detection](https://huggingface.co/tayyabimam/Deepfake)

## License

This project uses open-source models and libraries. Please check individual model licenses.

## Contributing

Feel free to improve the models, add more detection capabilities, or enhance the UI.