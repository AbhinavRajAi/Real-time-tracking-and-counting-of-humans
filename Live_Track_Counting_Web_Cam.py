import ultralytics
ultralytics.checks()

from IPython import display
display.clear_output()

import supervision as sv
import numpy as np
import cv2
from ultralytics import YOLO

# Load pre-trained YOLOv8 model
MODEL = "yolov8x.pt"
model = YOLO(MODEL)
model.fuse()


# dict mapping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# class_ids of interest - human
selected_classes = [0]

# Create BYTETracker instance
byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)


# Create instance of BoxAnnotator
box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Create instance of TraceAnnotator
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)

# Create LineZoneAnnotator instance
#line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

# Initialize counting variables
human_count = 0
tracked_ids = set()

# Define callback function for video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    global human_count, tracked_ids

    # Model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Only consider class id from selected_classes defined above
    detections = detections[np.isin(detections.class_id, selected_classes)]

    # Tracking detections using ByteTrack
    detections = byte_tracker.update_with_detections(detections)

    # Count new unique human IDs and create labels with tracking IDs
    labels = []
    for _, _, _, class_id, tracker_id in detections:
        # Counting unique humans
        if tracker_id not in tracked_ids:
            human_count += 1
            tracked_ids.add(tracker_id)

        # Create label with tracking ID
        labels.append(f"ID: {tracker_id}")

    # Annotate frame with detections and tracking IDs
    annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

    # Display human count on frame
    cv2.putText(annotated_frame, f'Human Count: {human_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return annotated_frame


# Open webcam for input
cap = cv2.VideoCapture(0)

# Process the live video stream from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = callback(frame, 0)
    cv2.imshow('Live Counting', processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()

