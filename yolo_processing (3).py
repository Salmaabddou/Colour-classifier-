import cv2
import requests
from ultralytics import YOLO

# Load YOLO model
model = YOLO("saiko_hankoshi/train6/weights/best.pt")
url = "http://10.112.235.161:8080/video"
#flask_url = "http://192.168.137.229:5000/receive_data"  # Update with your Raspberry Pi's IP address

# Open the video stream
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame")
        break

    # Perform object detection
    results = model.predict(source=frame)

    # Initialize a dictionary to store class centers
    class_centers = {}

    # Get the frame with bounding boxes and labels plotted
    frame_with_boxes = results[0].plot()

    # Extract classes and centers from results
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]  # Extract bounding box coordinates
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            # Get class label
            class_label = box.cls  # Assuming the class label is stored in 'cls'
            class_name = model.names[int(class_label)]  # Get class name from model

            # Store the center of the bounding box in the dictionary
            if class_name not in class_centers:
                class_centers[class_name] = []

            class_centers[class_name].append((int(x_center), int(y_center)))

            # Plot the center of the bounding box
            center_coordinates = (int(x_center), int(y_center))
            cv2.circle(frame_with_boxes, center_coordinates, 5, (0, 0, 255), -1)  # Red circle for center

    # Print the dictionary (optional)
    print(class_centers)

    # Send the dictionary to the Flask server
    # try:
    #     response = requests.post(flask_url, json=class_centers)
    #     if response.status_code == 200:
    #         print("Data sent successfully")
    #     else:
    #         print(f"Failed to send data. Status code: {response.status_code}")
    # except requests.exceptions.RequestException as e:
    #     print(f"Error sending data: {e}")

    # Display the resulting frame with detections and centers
    cv2.imshow('YOLO Detection', frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
