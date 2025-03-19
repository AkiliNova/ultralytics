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

# Load the YOLOv11x model
model_path = 'yolo11x.pt'
model = YOLO(model_path)

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
    if frame is None:
        return None
        
    # Make a copy of the frame to make it writable
    frame = np.array(frame, copy=True)
    
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
    
    return frame

# Create Gradio interface with tabs
with gr.Blocks(title="YOLOv11x Object Detection") as iface:
    gr.Markdown("# YOLOv11x Object Detection")
    
    with gr.Tabs():
        # Video Upload/URL Tab
        with gr.Tab("Video Processing"):
            gr.Markdown("Upload a video file or provide a video URL for object detection")
            with gr.Row():
                video_input = gr.Video(label="Input Video")
                video_url = gr.Textbox(label="Or enter video URL", placeholder="https://example.com/video.mp4")
            
            process_btn = gr.Button("Process Video")
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
                live_input = gr.Image(sources=["webcam"], streaming=True)
                live_output = gr.Image()
            
            live_input.stream(
                fn=live_detection,
                inputs=live_input,
                outputs=live_output,
                show_progress=False
            )

if __name__ == "__main__":
    iface.launch(share=True, server_name="0.0.0.0", server_port=3000)