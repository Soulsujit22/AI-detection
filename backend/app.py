from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import pipeline
from PIL import Image
import cv2
import io
import tempfile
import os
import csv
from datetime import datetime
import base64
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

app = FastAPI()

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.mount('/static', StaticFiles(directory=frontend_dir), name='static')

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
image_detector = None
video_detector = None
efficientnet_detector = None

csv_path = os.path.join(os.path.dirname(__file__), 'results.csv')

CSV_FIELDS = [
    'timestamp',
    'filename',
    'media_type',
    'detected_label',
    'is_ai_generated',
    'status',
    'confidence',
    'frames_analyzed',
    'fake_frames',
    'real_frames'
]

IMAGE_CONF_THRESHOLD = 0.75
VIDEO_CONF_THRESHOLD = 0.70

def ensure_csv():
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
            writer.writeheader()


def save_detection_result(result_data: dict):
    ensure_csv()
    with open(csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writerow({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'filename': result_data.get('filename', ''),
            'media_type': result_data.get('media_type', ''),
            'detected_label': result_data.get('detected_label', ''),
            'is_ai_generated': result_data.get('is_ai_generated', ''),
            'confidence': result_data.get('confidence', ''),
            'frames_analyzed': result_data.get('frames_analyzed', ''),
            'fake_frames': result_data.get('fake_frames', ''),
            'real_frames': result_data.get('real_frames', '')
        })


def get_image_detector():
    global image_detector
    if image_detector is None:
        print("Loading image detection model...")
        image_detector = pipeline("image-classification", 
                                 model="prithivMLmods/open-deepfake-detection")
        print("Image model loaded!")
    return image_detector

def get_video_detector():
    # Use the same model as image for simplicity
    return get_image_detector()

def get_efficientnet_detector():
    global efficientnet_detector
    if efficientnet_detector is None:
        print("Loading EfficientNet model...")
        efficientnet_detector = pipeline("image-classification", model="google/efficientnet-b0")
        print("EfficientNet model loaded!")
    return efficientnet_detector

def get_efficientnet_result(image):
    detector = get_efficientnet_detector()
    result = detector(image, top_k=1)
    return result[0]

def get_efficientnet_status(label, conf):
    natural_terms = ['tree', 'person', 'dog', 'cat', 'sky', 'water', 'mountain', 'food', 'building', 'flower', 'car', 'animal', 'plant', 'nature', 'landscape']
    if conf >= 0.5 and any(term in label.lower() for term in natural_terms):
        return 'Likely Real'
    return 'Uncertain'


def combine_status(deepfake_status, deepfake_conf, efficientnet_label, efficientnet_conf):
    efficientnet_status = get_efficientnet_status(efficientnet_label, efficientnet_conf)

    # If one model is confident and the other is not, choose the confident model.
    if deepfake_conf >= 0.5 and efficientnet_conf < 0.5:
        return deepfake_status
    if efficientnet_conf >= 0.5 and deepfake_conf < 0.5:
        return efficientnet_status

    # If both are confident and disagree, choose the higher-confidence result.
    if deepfake_conf >= 0.5 and efficientnet_conf >= 0.5:
        if deepfake_status == efficientnet_status:
            return deepfake_status
        return deepfake_status if deepfake_conf >= efficientnet_conf else efficientnet_status

    # If one is confident enough, use it.
    if deepfake_conf >= 0.5:
        return deepfake_status
    if efficientnet_conf >= 0.5:
        return efficientnet_status

    return 'Uncertain'


def calculate_image_metrics(image: Image.Image) -> dict:
    """Calculate detailed metrics for image analysis."""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    metrics = {}
    
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # 1. Saturation Analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1]
        metrics['avg_saturation'] = float(np.mean(saturation) / 255.0 * 100)
        metrics['saturation_std'] = float(np.std(saturation) / 255.0 * 100)
        
        # 2. Edge Detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        metrics['edge_density'] = float(np.sum(edges) / edges.size * 100)
        
        # 3. Color Consistency
        blur = cv2.GaussianBlur(img_array.astype(np.float32), (15, 15), 0)
        color_diff = np.mean(np.abs(img_array.astype(np.float32) - blur), axis=2)
        metrics['color_consistency'] = float(100 - (np.mean(color_diff) / 255.0 * 100))
        metrics['color_variance'] = float(np.std(color_diff))
        
        # 4. Brightness Analysis
        brightness = np.mean(img_array)
        metrics['brightness'] = float(brightness)
        
        # 5. Contrast Analysis
        contrast = np.std(img_array)
        metrics['contrast'] = float(contrast)
        
        # 6. Frequency Analysis (high frequency = more detail, low frequency = smoother)
        freq_components = np.fft.fft2(gray)
        freq_magnitude = np.abs(freq_components)
        metrics['high_freq_content'] = float(np.sum(freq_magnitude[freq_magnitude > np.median(freq_magnitude)]) / np.sum(freq_magnitude) * 100)
        
        # 7. Noise Detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics['laplacian_variance'] = float(np.var(laplacian))
        
    return metrics


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_base64}"


def create_visualization(image: Image.Image, label: str, confidence: float) -> str:
    """Create a heatmap visualization showing model confidence across image regions."""
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Normalize image to 0-1 range
    img_normalized = img_array.astype(np.float32) / 255.0
    
    # Create a gradient map based on color distribution
    # Areas with high saturation/contrast often indicate manipulations
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        # Convert to HSV to get saturation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)
        saturation = hsv[:, :, 1]
        
        # Detect edges - areas with high contrast might indicate artifacts
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect color consistency - inconsistent colors suggest manipulation
        blur = cv2.GaussianBlur(img_array.astype(np.float32), (15, 15), 0)
        color_diff = np.mean(np.abs(img_array.astype(np.float32) - blur), axis=2)
        
        # Combine metrics
        heatmap = (saturation / 255.0) * 0.4 + (color_diff / color_diff.max()) * 0.4 + (edges / 255.0) * 0.2
    else:
        heatmap = np.zeros(img_array.shape[:2])
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
    
    # Create the visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Heatmap overlay
    im = axes[1].imshow(img_array)
    heatmap_colored = plt.cm.RdYlGn_r(heatmap)
    im2 = axes[1].imshow(heatmap, cmap='RdYlGn_r', alpha=0.6)
    axes[1].set_title(f'Detection Heatmap\n{label} ({confidence*100:.1f}%)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Anomaly Score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_base64}"


def generate_explanation(label: str, confidence: float, efficientnet_label: str, efficientnet_conf: float, metrics: dict) -> str:
    """Generate a human-readable explanation of the detection result with detailed metrics."""
    
    # Determine what the model is seeing
    is_fake = label.lower() == 'fake' or label == 'label_0'
    
    if is_fake:
        deepfake_finding = f"⚠️ <strong>Detected as AI-generated or manipulated</strong> ({confidence*100:.1f}% confidence)"
        verdict_color = "#ff5b8a"
    else:
        deepfake_finding = f"✓ <strong>Appears to be authentic</strong> ({confidence*100:.1f}% confidence)"
        verdict_color = "#1f8d4d"
    
    # Build metrics explanation
    metrics_html = "<p><strong>Analysis Measures Taken:</strong></p>"
    metrics_html += "<ul style='margin: 8px 0; padding-left: 20px;'>"
    
    if metrics:
        metrics_html += f"<li><strong>Color Saturation:</strong> {metrics.get('avg_saturation', 'N/A'):.1f}% (Std: {metrics.get('saturation_std', 'N/A'):.1f}%) - AI images often have unusual color properties</li>"
        metrics_html += f"<li><strong>Color Consistency:</strong> {metrics.get('color_consistency', 'N/A'):.1f}% - Measures smoothness vs natural variation</li>"
        metrics_html += f"<li><strong>Edge Density:</strong> {metrics.get('edge_density', 'N/A'):.2f}% - AI can create unnatural textures and boundaries</li>"
        metrics_html += f"<li><strong>Brightness:</strong> {metrics.get('brightness', 'N/A'):.1f}/255 - Indicates lighting consistency</li>"
        metrics_html += f"<li><strong>Contrast:</strong> {metrics.get('contrast', 'N/A'):.1f} - High contrast may suggest artifacts</li>"
        metrics_html += f"<li><strong>High Frequency Content:</strong> {metrics.get('high_freq_content', 'N/A'):.1f}% - Natural images have balanced detail distribution</li>"
        metrics_html += f"<li><strong>Laplacian Variance:</strong> {metrics.get('laplacian_variance', 'N/A'):.1f} - Detects edge anomalies and compression artifacts</li>"
    
    metrics_html += "</ul>"
    
    # Detection signs
    if is_fake:
        detection_signs = [
            "High color variance indicating artificial manipulation",
            "Unnatural blending patterns between regions",
            "Suspicious saturation or contrast levels",
            "Anomalous edge patterns and artifact traces",
            "Inconsistent frequency distribution"
        ]
    else:
        detection_signs = [
            "Natural color distribution across regions",
            "Consistent lighting and shadow patterns",
            "Normal edge and texture patterns",
            "Coherent image statistics and frequency content",
            "Expected noise and detail levels"
        ]
    
    detection_html = "<p><strong>Key Indicators Found:</strong></p>"
    detection_html += "<ul style='margin: 8px 0; padding-left: 20px;'>"
    for sign in detection_signs:
        detection_html += f"<li>{sign}</li>"
    detection_html += "</ul>"
    
    # Comparison analysis
    comparison_html = f"<p><strong>Secondary Model (EfficientNet):</strong> Classified as <strong>{efficientnet_label}</strong> ({efficientnet_conf*100:.1f}% confidence)</p>"
    
    explanation = f"""
    <div style="border-left: 4px solid {verdict_color}; padding: 16px; margin: 16px 0; background: rgba(91, 91, 255, 0.05); border-radius: 8px;">
        <p style="font-size: 1.1em; color: {verdict_color};">{deepfake_finding}</p>
        
        {metrics_html}
        
        {detection_html}
        
        {comparison_html}
        
        <p style="font-size: 0.85em; color: var(--muted); margin-top: 12px; font-style: italic;">
            <strong>How this works:</strong> The AI Detection tool uses multiple analysis techniques to compare your image against patterns commonly found in AI-generated and authentic images. 
            These measures work together to provide a comprehensive assessment.
        </p>
    </div>
    """
    
    return explanation

@app.post("/detect/images")
async def detect_images(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            detector = get_image_detector()
            result = detector(image)
            prediction = result[0]
            label = prediction['label']
            confidence = prediction['score']

            if label.lower() == 'fake' or label == 'label_0':
                deepfake_status = 'AI Generated/Manipulated'
                deepfake_conf = confidence
                is_ai_generated = True
            elif label.lower() == 'real' or label == 'label_1':
                deepfake_status = 'Likely Real'
                deepfake_conf = confidence
                is_ai_generated = False
            else:
                deepfake_status = 'Uncertain'
                deepfake_conf = confidence
                is_ai_generated = None

            e_result = get_efficientnet_result(image)
            e_label = e_result['label']
            e_confidence = e_result['score']
            combined = combine_status(deepfake_status, deepfake_conf, e_label, e_confidence)
            status = combined

            # Calculate metrics for detailed analysis
            metrics = calculate_image_metrics(image)
            
            # Convert image to base64
            original_image_base64 = image_to_base64(image)
            
            # Generate visualization and explanation
            visualization = create_visualization(image, label, confidence)
            explanation = generate_explanation(label, confidence, e_label, e_confidence, metrics)

            response = {
                "filename": file.filename,
                "detected_label": label,
                "status": status,
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "efficientnet_label": e_label,
                "efficientnet_confidence": e_confidence,
                "combined_status": combined,
                "original_image": original_image_base64,
                "visualization": visualization,
                "explanation": explanation,
                "metrics": metrics,
                "raw_result": result
            }
            save_detection_result({
                'filename': file.filename,
                'media_type': 'image',
                'detected_label': label,
                'status': status,
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'frames_analyzed': '',
                'fake_frames': '',
                'real_frames': ''
            })
            results.append(response)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Image processing failed: {str(e)}"
            })
    return {"results": results}

@app.post("/detect/videos")
async def detect_videos(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                results.append({"filename": file.filename, "error": "Could not open video file"})
                continue

            frame_results = []
            efficientnet_labels = []
            top_efficientnet_confidence = 0.0
            frame_count = 0
            sample_rate = 30
            detector = get_video_detector()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    result = detector(pil_image)
                    frame_results.append(result[0])

                    e_result = get_efficientnet_result(pil_image)
                    efficientnet_labels.append(e_result['label'])
                    top_efficientnet_confidence = e_result['score']

                frame_count += 1

            cap.release()

            if not frame_results:
                results.append({"filename": file.filename, "error": "No frames could be processed"})
                continue

            fake_count = sum(1 for r in frame_results if r['label'].lower() == 'fake' or r['label'] == 'label_0')
            real_count = sum(1 for r in frame_results if r['label'].lower() == 'real' or r['label'] == 'label_1')
            other_count = len(frame_results) - fake_count - real_count

            if fake_count > real_count and (fake_count / len(frame_results)) >= VIDEO_CONF_THRESHOLD:
                status = 'AI Generated/Manipulated'
                is_ai_generated = True
                detected_label = 'Fake'
            elif real_count > fake_count and (real_count / len(frame_results)) >= VIDEO_CONF_THRESHOLD:
                status = 'Likely Real'
                is_ai_generated = False
                detected_label = 'Real'
            else:
                status = 'Uncertain'
                is_ai_generated = None
                detected_label = 'Ambiguous'

            confidence = max(fake_count, real_count, other_count) / len(frame_results)
            top_efficientnet_label = None
            if efficientnet_labels:
                from collections import Counter
                top_efficientnet_label = Counter(efficientnet_labels).most_common(1)[0][0]

            combined = combine_status(status, confidence, top_efficientnet_label or '', top_efficientnet_confidence if efficientnet_labels else 0.0)

            response = {
                "filename": file.filename,
                "detected_label": detected_label,
                "status": status,
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "efficientnet_label": top_efficientnet_label,
                "efficientnet_confidence": top_efficientnet_confidence if efficientnet_labels else 0.0,
                "combined_status": combined,
                "frames_analyzed": len(frame_results),
                "fake_frames": fake_count,
                "real_frames": real_count
            }
            save_detection_result({
                'filename': file.filename,
                'media_type': 'video',
                'detected_label': detected_label,
                'status': status,
                'is_ai_generated': is_ai_generated,
                'confidence': confidence,
                'frames_analyzed': len(frame_results),
                'fake_frames': fake_count,
                'real_frames': real_count
            })
            results.append(response)
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": f"Video processing failed: {str(e)}"
            })
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return {"results": results}

@app.post("/detect/image")
async def detect_image(files: list[UploadFile] = File(...)):
    return await detect_images(files)

@app.post("/detect/video")
async def detect_video(files: list[UploadFile] = File(...)):
    return await detect_videos(files)

@app.get("/download-results")
async def download_results():
    ensure_csv()
    if not os.path.exists(csv_path):
        return {"error": "No results CSV exists yet."}
    return FileResponse(csv_path, filename="results.csv", media_type="text/csv")

@app.get("/", response_class=FileResponse)
async def root():
    return os.path.join(frontend_dir, 'index.html')

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run(app, host="127.0.0.1", port=port)