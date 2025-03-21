import gradio as gr
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import requests
from urllib.parse import urlparse
import time
import threading
from datetime import datetime
import logging
from gradio.components import State

# Configure logging to reduce OpenCV FFMPEG warnings
logging.getLogger("ultralytics").setLevel(logging.WARNING)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000|max_delay;500000|reorder_queue_size;0|stimeout;1000000"

# Load the YOLOv11x model
model_path = 'yolo11x.pt'
model = YOLO(model_path)
# Set verbose level to reduce logs
model.verbose = False

# Pushbullet configuration
PUSHBULLET_API_KEY = "o.Lmbo70vjPNtmASSwvNIJ6yTiVyd83vKV"
last_notification_time = 0
NOTIFICATION_COOLDOWN = 60  # seconds

# RTSP stream configuration
RTSP_URL = 'rtsp://admin:akilicamera@154.70.45.143:554/mode=real&idc=1&ids=1'

# Global variables for stream handling
stream_active = False
current_frame = None
frame_lock = threading.Lock()
detection_logs = []

def send_pushbullet_notification(title, message):
    """Send notification via Pushbullet"""
    url = "https://api.pushbullet.com/v2/pushes"
    headers = {
        "Access-Token": PUSHBULLET_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "type": "note",
        "title": title,
        "body": message
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send notification: {e}")

def add_detection_log(message):
    """Add a detection log with timestamp"""
    global detection_logs
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detection_logs.insert(0, f"[{timestamp}] {message}")
    # Keep only last 100 logs
    detection_logs = detection_logs[:100]
    return "\n".join(detection_logs)

def process_rtsp_stream():
    """Process RTSP stream in a separate thread"""
    global current_frame, stream_active, last_notification_time
    
    try:
        # Set OpenCV backend to FFMPEG with TCP transport
        stream_url = RTSP_URL
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        
        # Configure stream settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS to reduce load
        
        if not cap.isOpened():
            add_detection_log("Failed to connect to RTSP stream")
            return
            
        add_detection_log("Successfully connected to RTSP stream")
        frame_count = 0
        retry_count = 0
        max_retries = 3
        last_frame_time = time.time()
        
        while stream_active:
            try:
                ret, frame = cap.read()
                current_time = time.time()
                
                # Check for timeout
                if current_time - last_frame_time > 5:  # 5 seconds timeout
                    add_detection_log("Stream timeout detected, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    last_frame_time = current_time
                    continue
                
                if not ret:
                    retry_count += 1
                    add_detection_log(f"Failed to read frame (attempt {retry_count}/{max_retries})...")
                    if retry_count >= max_retries:
                        add_detection_log("Max retries reached, reconnecting stream...")
                        cap.release()
                        time.sleep(2)
                        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                        retry_count = 0
                    continue
                
                last_frame_time = current_time
                retry_count = 0  # Reset retry count on successful frame read
                frame_count += 1
                
                if frame_count % 3 != 0:  # Process every 3rd frame
                    continue
                
                # Ensure frame is valid
                if frame is None or frame.size == 0:
                    continue
                
                # Process frame with YOLO (with minimal logging)
                with torch.no_grad():
                    results = model(frame, verbose=False)
                
                # Track if cup was detected
                cup_detected = False
                cup_confidence = 0.0
                
                # Draw boxes on frame
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = model.names[cls]
                        
                        # Draw box and label for all objects
                        color = (0, 255, 0) if label.lower() == 'cup' else (255, 0, 0)  # Green for cups, red for others
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{label} {conf:.2%}', (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        # Add detection log for all objects
                        add_detection_log(f"{label} detected with {conf:.2%} confidence")
                        
                        # Check if the detected object is a cup
                        if label.lower() == 'cup':
                            cup_detected = True
                            cup_confidence = conf
                
                # Send notification if cup is detected
                if cup_detected:
                    current_time = time.time()
                    if current_time - last_notification_time > NOTIFICATION_COOLDOWN:
                        send_pushbullet_notification(
                            "Cup Detected!",
                            f"A cup was detected with {cup_confidence:.2%} confidence."
                        )
                        add_detection_log("Pushbullet notification sent")
                        last_notification_time = current_time
                
                # Update current frame
                with frame_lock:
                    current_frame = frame.copy()
                
            except Exception as e:
                add_detection_log(f"Stream error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()
        add_detection_log("Stream stopped")

def get_current_frame():
    """Get the current processed frame for Gradio"""
    with frame_lock:
        if current_frame is None:
            # Return a black frame with text
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, 'Waiting for stream...', (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return frame
        return current_frame.copy()  # Return a copy to prevent race conditions

def start_stream():
    """Start the RTSP stream processing"""
    global stream_active
    if not stream_active:
        stream_active = True
        thread = threading.Thread(target=process_rtsp_stream)
        thread.daemon = True
        thread.start()
        return add_detection_log("Starting stream...")
    return add_detection_log("Stream is already running")

def stop_stream():
    """Stop the RTSP stream processing"""
    global stream_active
    stream_active = False
    return add_detection_log("Stream stopping...")

def download_video(url):
    """Download video from URL to temporary file"""
    try:
        # Create temp dir if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        
        # Get filename from URL or use default
        filename = os.path.basename(urlparse(url).path)
        if not filename:
            filename = 'video.mp4'
        
        temp_path = os.path.join(temp_dir, filename)
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        return temp_path
    except Exception as e:
        raise gr.Error(f"Failed to download video: {str(e)}")

def process_video(video_input, progress=gr.Progress()):
    """Process video file or URL"""
    try:
        # If input is URL, download it first
        if isinstance(video_input, str) and (video_input.startswith('http://') or video_input.startswith('https://')):
            video_path = download_video(video_input)
        else:
            video_path = video_input
            
        # Create a temporary file for the output video
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'output.mp4')
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        for frame_idx in progress.tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Run inference on frame
            results = model(frame)
            
            # Draw boxes on frame
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write the frame
            out.write(frame)
        
        # Release everything
        cap.release()
        out.release()
        
        return output_path
    except Exception as e:
        raise gr.Error(f"Error processing video: {str(e)}")

def live_detection(frame):
    """Process live webcam feed"""
    global last_notification_time
    
    if frame is None:
        return None
        
    # Make a copy of the frame to make it writable
    frame = np.array(frame, copy=True)
    
    # Run inference on frame with minimal logging
    with torch.no_grad():
        results = model(frame, verbose=False)
    
    # Track if cup was detected in this frame
    cup_detected = False
    cup_confidence = 0.0
    
    # Draw boxes on frame
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Send notification if cup is detected (with cooldown)
    if cup_detected:
        current_time = time.time()
        if current_time - last_notification_time > NOTIFICATION_COOLDOWN:
            send_pushbullet_notification(
                "Cup Detected!",
                f"A cup was detected with {cup_confidence:.2%} confidence."
            )
            last_notification_time = current_time
    
    return frame

# Create Gradio interface with tabs
with gr.Blocks(
    title="YOLOv11x Object Detection",
    css="""
        .gradio-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 !important;
        }
        .output-image, .input-image {
            height: 720px !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        .output-image img, .input-image img {
            max-height: 100% !important;
            width: 100% !important;
            object-fit: contain !important;
        }
        .tabs {
            width: 100% !important;
            max-width: 100% !important;
        }
        .tabitem {
            width: 100% !important;
            max-width: 100% !important;
        }
        .row {
            width: 100% !important;
            max-width: 100% !important;
        }
        .column {
            width: 100% !important;
            max-width: 100% !important;
        }
        .video-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        .video-container video {
            width: 100% !important;
            max-width: 100% !important;
        }
        .textbox {
            width: 100% !important;
            max-width: 100% !important;
        }
        .container {
            width: 100% !important;
            max-width: 100% !important;
        }
        .wrap {
            width: 100% !important;
            max-width: 100% !important;
        }
        .wrap.svelte-1gigp65 {
            width: 100% !important;
            max-width: 100% !important;
        }
    """
) as iface:
    gr.Markdown("# YOLOv11x Object Detection")
    
    with gr.Tabs():
        # Video Upload/URL Tab
        with gr.Tab("Video Processing"):
            gr.Markdown("Upload a video file or provide a video URL for object detection")
            with gr.Row():
                video_input = gr.Video(label="Input Video", scale=1)
                video_url = gr.Textbox(
                    label="Or enter video URL",
                    placeholder="https://example.com/video.mp4",
                    scale=1
                )
            
            process_btn = gr.Button("Process Video", variant="primary")
            video_output = gr.Video(label="Processed Video")
            
            process_btn.click(
                fn=process_video,
                inputs=[video_input],
                outputs=video_output
            )
            
            video_url.submit(
                fn=process_video,
                inputs=[video_url],
                outputs=video_output
            )
        
        # Live Detection Tab
        with gr.Tab("Live Detection"):
            gr.Markdown("Use your webcam for real-time object detection")
            with gr.Row():
                live_input = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    scale=1,
                    height=720
                )
                live_output = gr.Image(
                    label="Detection Output",
                    scale=1,
                    height=720
                )
            
            live_input.stream(
                fn=live_detection,
                inputs=live_input,
                outputs=live_output,
                show_progress=False
            )
            
        # RTSP Stream Tab
        with gr.Tab("RTSP Stream"):
            gr.Markdown("Live RTSP camera feed with object detection")
            
            with gr.Row():
                start_btn = gr.Button("Start Stream", variant="primary", scale=1)
                stop_btn = gr.Button("Stop Stream", variant="stop", scale=1)
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1)
            
            with gr.Row():
                with gr.Column(scale=2):
                    camera_feed = gr.Image(
                        label="Camera Feed",
                        interactive=False,
                        height=720
                    )
                with gr.Column(scale=1):
                    detection_log = gr.Textbox(
                        label="Detection Logs",
                        value="Ready to start...",
                        lines=20,
                        max_lines=20,
                        interactive=False
                    )
            
            def update_feed_and_logs():
                """Update both feed and logs"""
                frame = get_current_frame() if stream_active else None
                logs = "\n".join(detection_logs)
                return frame, logs

            # Update the stream handling
            start_btn.click(
                fn=start_stream,
                outputs=detection_log
            )
            
            stop_btn.click(
                fn=stop_stream,
                outputs=detection_log
            )

            # Add refresh button for updates
            refresh_btn.click(
                fn=update_feed_and_logs,
                outputs=[camera_feed, detection_log]
            )

            # Add auto-refresh using JavaScript
            gr.Markdown("""
                <script>
                    function setupAutoRefresh() {
                        // Find the refresh button by its emoji and text content
                        function findRefreshButton() {
                            const buttons = document.querySelectorAll('button');
                            for (const btn of buttons) {
                                if (btn.textContent.includes('🔄')) {
                                    return btn;
                                }
                            }
                            return null;
                        }

                        // Set up the interval
                        let refreshInterval;
                        
                        // Function to start auto-refresh
                        function startAutoRefresh() {
                            if (!refreshInterval) {
                                const refreshBtn = findRefreshButton();
                                if (refreshBtn) {
                                    refreshInterval = setInterval(() => {
                                        refreshBtn.click();
                                    }, 200);  // Refresh every 200ms
                                }
                            }
                        }
                        
                        // Function to stop auto-refresh
                        function stopAutoRefresh() {
                            if (refreshInterval) {
                                clearInterval(refreshInterval);
                                refreshInterval = null;
                            }
                        }
                        
                        // Watch for the start/stop buttons
                        const observer = new MutationObserver(() => {
                            const startBtn = document.querySelector('button:contains("Start Stream")');
                            const stopBtn = document.querySelector('button:contains("Stop Stream")');
                            
                            if (startBtn && stopBtn) {
                                startBtn.addEventListener('click', startAutoRefresh);
                                stopBtn.addEventListener('click', stopAutoRefresh);
                                observer.disconnect();
                            }
                        });
                        
                        observer.observe(document.body, {
                            childList: true,
                            subtree: true
                        });
                    }
                    
                    // Run setup when the page loads
                    if (document.readyState === 'complete') {
                        setupAutoRefresh();
                    } else {
                        window.addEventListener('load', setupAutoRefresh);
                    }
                </script>
            """)

if __name__ == "__main__":
    iface.queue()  # Enable queuing for better performance
    iface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=3000,
        favicon_path=None,
        show_error=True,
        quiet=True
    )