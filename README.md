# teddy-detector
# Teddy Bear Detection System

A real-time web application that detects teddy bears in images using YOLOv8 and FastAPI.

## Features

- Upload images through drag-and-drop or file selection
- Real-time teddy bear detection using YOLOv8
- Visual alerts when teddy bears are detected
- Statistics tracking for detections and false alarms
- Interactive graphical representation of detection history
- Mobile-friendly responsive design

## Tech Stack

- **Backend**: FastAPI
- **ML Model**: YOLOv8
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, Pillow
- **Deployment**: Render

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn app:app --reload
```

3. Access the web interface at `http://localhost:8000`

## Project Structure

- `app.py` - Main FastAPI application
- `best.pt` - Trained YOLOv8 model
- `requirements.txt` - Python dependencies
- `render.yaml` - Render deployment configuration

## Usage

1. Open the web interface
2. Upload an image by dragging and dropping or clicking the upload area
3. View detection results and alerts
4. Check statistics by clicking the Statistics button

## Deployment

This application is configured for deployment on Render. The `render.yaml` file contains the necessary deployment configuration. 
