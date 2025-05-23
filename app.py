from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import logging
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Create necessary directories
os.makedirs("static", exist_ok=True)

# Initialize statistics file
STATS_FILE = "detection_stats.json"
if not os.path.exists(STATS_FILE):
    with open(STATS_FILE, "w") as f:
        json.dump({"detections": [], "total_detections": 0, "total_false_alarms": 0}, f)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model
try:
    logger.info("Loading YOLO model...")
    model_path = "best.pt"  # Model file in root directory
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = YOLO(model_path)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {str(e)}")
    raise

@app.get("/stats")
async def get_stats():
    try:
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
        return stats
    except Exception as e:
        logger.error(f"Error reading stats: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Teddy Bear Detection</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --primary-blue: #1a73e8;
                --light-blue: #e8f0fe;
                --hover-blue: #1557b0;
                --border-blue: #4285f4;
            }
            
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 30px;
                background-color: #f8f9fa;
            }
            
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid #e1e4e8;
                position: relative;
            }
            
            .stats-button {
                position: absolute;
                top: 20px;
                right: 20px;
                background-color: var(--primary-blue);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: background-color 0.3s;
            }
            
            .stats-button:hover {
                background-color: var(--hover-blue);
            }
            
            .modal {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 1000;
            }
            
            .modal-content {
                background-color: white;
                margin: 0;
                padding: 30px;
                width: 100%;
                height: 100%;
                position: relative;
                overflow-y: auto;
                box-sizing: border-box;
            }
            
            .close-button {
                position: fixed;
                top: 20px;
                right: 30px;
                font-size: 24px;
                cursor: pointer;
                color: #666;
                z-index: 1010;
                background-color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .close-button:hover {
                background-color: #f0f0f0;
            }
            
            .stats-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin: 20px 0;
            }
            
            .chart-container {
                margin: 30px 0;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e1e4e8;
                height: 400px;
            }
            
            .detection-list {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #e1e4e8;
                border-radius: 8px;
                background-color: white;
            }

            .stats-header {
                margin-bottom: 30px;
                text-align: center;
            }

            .stats-section {
                margin-bottom: 40px;
            }
            
            .stat-card {
                background-color: var(--light-blue);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            
            .stat-number {
                font-size: 2em;
                color: var(--primary-blue);
                font-weight: bold;
            }
            
            .detection-item {
                padding: 10px;
                border-bottom: 1px solid #e1e4e8;
                display: flex;
                justify-content: space-between;
            }
            
            .detection-item:last-child {
                border-bottom: none;
            }
            
            #dropZone {
                border: 2px dashed var(--border-blue);
                border-radius: 8px;
                padding: 40px 20px;
                text-align: center;
                margin: 20px 0;
                cursor: pointer;
                background-color: var(--light-blue);
                transition: all 0.3s ease;
                color: var(--primary-blue);
                font-size: 1.1em;
            }
            
            #dropZone:hover {
                background-color: #f0f7ff;
                border-color: var(--primary-blue);
            }
            
            #dropZone.dragover {
                background-color: #f0f7ff;
                border-color: var(--primary-blue);
                transform: scale(1.02);
            }
            
            #resultImage {
                max-width: 100%;
                margin-top: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
                color: var(--primary-blue);
                font-weight: 500;
            }
            
            .loading::after {
                content: '';
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-left: 10px;
                border: 3px solid var(--light-blue);
                border-top: 3px solid var(--primary-blue);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            #error {
                color: #d93025;
                display: none;
                margin: 15px 0;
                text-align: center;
                padding: 12px;
                background-color: #fce8e6;
                border-radius: 8px;
                border: 1px solid #fad2cf;
            }
            
            #message {
                color: #6c757d;
                display: none;
                margin: 15px 0;
                text-align: center;
                padding: 12px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #dee2e6;
                font-weight: 500;
                font-size: 1.2em;
                animation: bounceIn 0.5s ease-out;
            }
            
            @keyframes bounceIn {
                0% {
                    transform: scale(0.3);
                    opacity: 0;
                }
                50% {
                    transform: scale(1.05);
                    opacity: 0.8;
                }
                100% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            .result-container {
                position: relative;
            }

            /* Alert styles */
            .alert {
                display: none;
                background-color: #ff4444;
                color: white;
                padding: 15px 20px;
                border-radius: 8px;
                margin: 20px 0;
                text-align: center;
                font-weight: bold;
                font-size: 1.2em;
                position: relative;
                animation: alertPulse 2s infinite;
                box-shadow: 0 4px 12px rgba(255, 68, 68, 0.2);
            }

            @keyframes alertPulse {
                0% { background-color: #ff4444; }
                50% { background-color: #cc0000; }
                100% { background-color: #ff4444; }
            }

            .alert-icon {
                display: inline-block;
                margin-right: 10px;
                animation: alertBlink 1s infinite;
            }

            @keyframes alertBlink {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }

            .detection-overlay {
                display: none;
            }

            .history-stats {
                position: fixed;
                bottom: 20px;
                right: 20px;
                background-color: rgba(255, 255, 255, 0.95);
                padding: 15px 20px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border: 1px solid #e1e4e8;
                font-size: 0.9em;
                color: #666;
                max-width: 300px;
                backdrop-filter: blur(5px);
                transition: opacity 0.3s;
                z-index: 100;
            }
            
            .history-stats:hover {
                opacity: 1;
            }
            
            .history-stats.with-detection {
                border-left: 4px solid var(--primary-blue);
            }
            
            .history-stats.with-false-alarm {
                border-left: 4px solid #ff9800;
            }
            
            .history-icon {
                margin-right: 8px;
                font-size: 1.1em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Teddy Bear Detection</h1>
            <button class="stats-button" onclick="openStats()">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M23 6l-9.5 9.5-5-5L1 18"/>
                    <path d="M17 6h6v6"/>
                </svg>
                Statistics
            </button>
            <div id="dropZone">
                <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <p style="margin: 10px 0 0 0;">Drop an image here or click to upload</p>
                <input type="file" id="fileInput" style="display: none;" accept="image/*">
            </div>
            <div class="loading" id="loading">Processing your image</div>
            <div id="error"></div>
            <div id="message"></div>
            <div class="alert" id="alert">
                <span class="alert-icon">⚠️</span>
                <span id="alertText"></span>
            </div>
            <div class="result-container">
                <img id="resultImage" style="display: none;">
            </div>
        </div>

        <!-- Statistics Modal -->
        <div id="statsModal" class="modal">
            <div class="modal-content">
                <span class="close-button" onclick="closeStats()">&times;</span>
                <div class="stats-container">
                    <div class="stats-header">
                        <h1>Surveillance Statistics</h1>
                    </div>
                    
                    <div class="stats-section">
                        <div class="stats-grid">
                            <div class="stat-card">
                                <div class="stat-number" id="totalDetections">0</div>
                                <div>Teddy Bears Detected</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-number" id="totalFalseAlarms">0</div>
                                <div>False Alarms</div>
                            </div>
                        </div>
                    </div>

                    <div class="stats-section">
                        <h2>Detection History</h2>
                        <div class="chart-container">
                            <canvas id="detectionChart"></canvas>
                        </div>
                    </div>

                    <div class="stats-section">
                        <h2>Recent Detections</h2>
                        <div class="detection-list" id="detectionList"></div>
                    </div>
                </div>
            </div>
        </div>

        <div id="historyStats" class="history-stats" style="display: none;">
            <span class="history-icon">📊</span>
            <span id="historyText"></span>
        </div>

        <script>
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const loading = document.getElementById('loading');
            const resultImage = document.getElementById('resultImage');
            const errorDiv = document.getElementById('error');
            const messageDiv = document.getElementById('message');
            const alert = document.getElementById('alert');
            const alertText = document.getElementById('alertText');

            dropZone.addEventListener('click', () => fileInput.click());
            
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
            
            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('dragover');
            });
            
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file) processFile(file);
            });
            
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) processFile(file);
            });

            async function processFile(file) {
                loading.style.display = 'block';
                resultImage.style.display = 'none';
                errorDiv.style.display = 'none';
                messageDiv.style.display = 'none';
                alert.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/detect/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const data = await response.json();
                        if (data.error) {
                            errorDiv.textContent = 'Error: ' + data.error;
                            errorDiv.style.display = 'block';
                        } else {
                            resultImage.src = 'data:image/jpeg;base64,' + data.image;
                            resultImage.style.display = 'block';
                            
                            if (data.teddy_detected) {
                                const count = data.teddy_count;
                                alert.style.display = 'block';
                                alertText.textContent = `ALERT: ${count} Teddy Bear${count > 1 ? 's' : ''} Detected!`;
                                updateHistoryStats(true);
                            } else if (data.message) {
                                messageDiv.textContent = data.message;
                                messageDiv.style.display = 'block';
                                updateHistoryStats(false);
                            }
                        }
                    } else {
                        const error = await response.json();
                        errorDiv.textContent = 'Error: ' + (error.detail || 'Failed to process image');
                        errorDiv.style.display = 'block';
                    }
                } catch (error) {
                    errorDiv.textContent = 'Error: ' + error.message;
                    errorDiv.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                }
            }

            let detectionChart = null;

            function processDetectionData(stats) {
                // Process data by day
                const detectionsByDay = {};
                const falseAlarmsByDay = {};
                
                stats.detections.forEach(detection => {
                    const date = new Date(detection.timestamp).toLocaleDateString();
                    if (!detectionsByDay[date]) {
                        detectionsByDay[date] = 0;
                        falseAlarmsByDay[date] = 0;
                    }
                    
                    if (detection.result.includes("No teddy bears detected")) {
                        falseAlarmsByDay[date]++;
                    } else {
                        // Extract number of detections from the result string
                        const match = detection.result.match(/Detected (\d+) teddy/);
                        if (match) {
                            detectionsByDay[date] += parseInt(match[1]);
                        }
                    }
                });
                
                // Sort dates
                const dates = Object.keys(detectionsByDay).sort();
                
                return {
                    labels: dates,
                    detections: dates.map(date => detectionsByDay[date]),
                    falseAlarms: dates.map(date => falseAlarmsByDay[date])
                };
            }

            function updateChart(stats) {
                const chartData = processDetectionData(stats);
                
                if (detectionChart) {
                    detectionChart.destroy();
                }
                
                const ctx = document.getElementById('detectionChart').getContext('2d');
                detectionChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: chartData.labels,
                        datasets: [
                            {
                                label: 'Teddy Bears Detected',
                                data: chartData.detections,
                                backgroundColor: 'rgba(26, 115, 232, 0.5)',
                                borderColor: 'rgba(26, 115, 232, 1)',
                                borderWidth: 1
                            },
                            {
                                label: 'False Alarms',
                                data: chartData.falseAlarms,
                                backgroundColor: 'rgba(255, 152, 0, 0.5)',
                                borderColor: 'rgba(255, 152, 0, 1)',
                                borderWidth: 1
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: true,
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    stepSize: 1
                                },
                                title: {
                                    display: true,
                                    text: 'Number of Events'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Date'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                position: 'top'
                            }
                        }
                    }
                });
            }
            
            async function openStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    document.getElementById('totalDetections').textContent = stats.total_detections;
                    document.getElementById('totalFalseAlarms').textContent = stats.total_false_alarms;
                    
                    // Update chart
                    updateChart(stats);
                    
                    const detectionList = document.getElementById('detectionList');
                    detectionList.innerHTML = '';
                    
                    stats.detections.slice().reverse().forEach(detection => {
                        const item = document.createElement('div');
                        item.className = 'detection-item';
                        item.innerHTML = `
                            <span>${detection.result}</span>
                            <span>${new Date(detection.timestamp).toLocaleString()}</span>
                        `;
                        detectionList.appendChild(item);
                    });
                    
                    document.getElementById('statsModal').style.display = 'block';
                } catch (error) {
                    console.error('Error fetching stats:', error);
                }
            }
            
            function closeStats() {
                document.getElementById('statsModal').style.display = 'none';
            }
            
            // Close modal when clicking outside
            window.onclick = function(event) {
                const modal = document.getElementById('statsModal');
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            }

            async function updateHistoryStats(teddy_detected) {
                const response = await fetch('/stats');
                const stats = await response.json();
                
                const historyStats = document.getElementById('historyStats');
                const historyText = document.getElementById('historyText');
                
                if (teddy_detected) {
                    historyStats.className = 'history-stats with-detection';
                    historyText.textContent = `⚠️ Security Alert: ${stats.total_detections} teddy bear intrusions recorded in the surveillance period`;
                } else {
                    historyStats.className = 'history-stats with-false-alarm';
                    historyText.textContent = `System Status: ${stats.total_false_alarms} false alarms during surveillance period`;
                }
                
                historyStats.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """

def count_recent_detections(stats, is_detection=True, days=5):
    """Count detections or false alarms in the last specified number of days."""
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(days=days)
    
    count = 0
    for detection in stats["detections"]:
        try:
            detection_time = datetime.fromisoformat(detection["timestamp"])
            if detection_time >= cutoff_time:
                # Specifically check for the detection format
                is_false_alarm = detection["result"].startswith("No teddy bears detected")
                if (not is_detection and is_false_alarm) or (is_detection and not is_false_alarm):
                    count += 1
        except (ValueError, KeyError):
            continue
    
    return count

def calculate_date_range(stats):
    """Calculate the date range of detections and return days span."""
    if not stats["detections"]:
        return 0
    
    dates = [datetime.fromisoformat(d["timestamp"]) for d in stats["detections"]]
    earliest = min(dates)
    latest = max(dates)
    days_span = (latest - earliest).days + 1  # +1 to include both start and end days
    return days_span

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)
        
        logger.info("Running YOLOv8 inference...")
        # Run YOLOv8 inference
        results = model(image)
        logger.info("Inference complete")
        
        # Update statistics
        with open(STATS_FILE, "r") as f:
            stats = json.load(f)
        
        detection_time = datetime.now().isoformat()
        teddy_count = len(results[0].boxes)
        
        if teddy_count == 0:
            logger.info("No teddy bears detected in the image")
            stats["total_false_alarms"] += 1
            detection_result = "No teddy bears detected - False alarm, oopsie! 🙈"
            stats["detections"].append({
                "result": detection_result,
                "timestamp": detection_time
            })
            result_image = image
            message = detection_result
        else:
            stats["total_detections"] += 1
            detection_result = f"Detected {teddy_count} teddy bear(s)"
            stats["detections"].append({
                "result": detection_result,
                "timestamp": detection_time
            })
            result_image = results[0].plot()
            border_size = 10
            result_image = cv2.copyMakeBorder(
                result_image,
                border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 255)
            )
            message = f"⚠️ {teddy_count} Teddy Bear{'s' if teddy_count > 1 else ''} Detected!"
        
        # Keep only the last 100 detections
        stats["detections"] = stats["detections"][-100:]
        
        # Save updated statistics
        with open(STATS_FILE, "w") as f:
            json.dump(stats, f)
        
        # Convert the image to base64
        is_success, buffer = cv2.imencode(".jpg", result_image)
        if not is_success:
            logger.error("Failed to encode result image")
            return JSONResponse(content={"error": "Failed to encode result image"}, status_code=500)
            
        img_str = base64.b64encode(buffer).decode()
        logger.info("Successfully processed image")
        
        # Return appropriate response
        if teddy_count == 0:
            return {
                "image": img_str,
                "message": message,
                "teddy_detected": False
            }
        else:
            return {
                "image": img_str,
                "teddy_detected": True,
                "teddy_count": teddy_count,
                "message": message
            }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500) 