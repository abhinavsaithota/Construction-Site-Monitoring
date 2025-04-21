import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

# Load YOLOv8 Helmet Detection Model
model = YOLO("best.pt")
image_count = 0

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator()

# Input video path
video_path = r"C:\Users\thota\OneDrive\Desktop\Track 3D\Construction_site_monitoring\AI_ConstructionSiteMonitoring-main\data\input.mp4"

# Output directory and video file
output_dir = r"C:\Users\thota\OneDrive\Desktop\Track 3D\Construction_site_monitoring\AI_ConstructionSiteMonitoring-main\outputs"
output_video = os.path.join(output_dir, "processed_video.mp4")

# Debug path check
print(f"Attempting to open video file at: {video_path}")
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found at {video_path}")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)


# Callback function to process frames
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global image_count
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Generate labels
    labels = [
        model.model.names[class_id]
        for class_id in detections.class_id
    ]

    # Draw boxes and labels
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # Save cropped detections
    for xyxy in detections.xyxy:
        cropped_image = sv.crop_image(image=frame, xyxy=xyxy)
        image_name = f"crop_{image_count}.png"
        image_path = os.path.join(output_dir, image_name)
        cv2.imwrite(image_path, cropped_image)
        image_count += 1

    return frame


# Process the video
sv.process_video(
    source_path=video_path,
    target_path=output_video,
    callback=callback
)

print(f"Video processed successfully. Output saved to: {output_video}")
