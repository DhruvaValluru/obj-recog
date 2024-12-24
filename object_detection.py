import cv2
import numpy as np

# Load the pre-trained model
model_path = "/Users/dvalluru/Downloads/object_detection/MobileNetSSD_deploy.caffemodel"
config_path = "/Users/dvalluru/Downloads/object_detection/MobileNetSSD_deploy.prototxt"

# Load model and configuration
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Define class labels MobileNet SSD detects
class_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", 
                "bus", "car", "cat", "chair", "cow", "diningtable", "dog", 
                "horse", "motorbike", "person", "pottedplant", "sheep", 
                "sofa", "train", "tvmonitor"]

# Start video capture (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Pass the blob through the network
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            label = class_labels[class_id]
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
