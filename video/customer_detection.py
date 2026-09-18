import cv2
from ultralytics import YOLO
import pandas as pd

# Load YOLO model
model = YOLO("yolov8n.pt")

# Input video
video_path = "video/retail_store.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Store customer analytics
customer_data = []

# Frame counter
frame_number = 0
previous_positions = {}
entered_ids = set()
exited_ids = set()

total_entered = 0
total_exited = 0
while cap.isOpened():

    # Read video frame
    ret, frame = cap.read()

    if not ret:
        break

    # Increase frame number
    frame_number += 1

    # Detect and track people
    results = model.track(
        frame,
        persist=True,
        classes=[0]
    )
    # Counting line position
    LINE_Y = 650
    # Count people in current frame
    person_count = 0

    # Store tracking IDs
    person_ids = []

    # Process detection results
    for result in results:

        # Check whether tracking IDs exist
        if result.boxes.id is not None:

            # Get boxes and tracking IDs
            for box, track_id in zip(
                result.boxes,
                result.boxes.id
            ):

                # Get class ID
                class_id = int(box.cls[0])

                # Class 0 = person
                if class_id == 0:

                    person_count += 1

                    person_ids.append(
                        int(track_id)
                    )
                     # Get customer ID
                    person_id = int(track_id)

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Find center of the person
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Check previous position of this customer
                    if person_id in previous_positions:

                        previous_y = previous_positions[person_id]

                        # Customer crossed the line from above to below
                        if previous_y < LINE_Y and center_y >= LINE_Y:

                            if person_id not in entered_ids:
                                entered_ids.add(person_id)
                                total_entered += 1

                        # Customer crossed the line from below to above
                        elif previous_y > LINE_Y and center_y <= LINE_Y:

                            if person_id not in exited_ids:
                                exited_ids.add(person_id)
                                total_exited += 1

                    # Save current position
                    previous_positions[person_id] = center_y
    # Draw detection boxes
    annotated_frame = results[0].plot()
    # Draw counting line
    cv2.line(
        annotated_frame,
        (0, LINE_Y),
        (annotated_frame.shape[1], LINE_Y),
        (0, 0, 255),
        3
    )

    # Display entry and exit counts
    cv2.putText(
        annotated_frame,
        f"Entered: {total_entered}",
        (20, annotated_frame.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Exited: {total_exited}",
        (20, annotated_frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )
    # Save customer count
    customer_data.append({
        "Frame": frame_number,
        "Customer_Count": person_count
    })

    # Display customer count
    cv2.putText(
        annotated_frame,
        f"Customers Detected: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display tracking IDs
    y_position = 80

    for person_id in person_ids:

        cv2.putText(
            annotated_frame,
            f"Customer ID: {person_id}",
            (20, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        y_position += 25

    # Show video
    cv2.imshow(
        "Retail Customer Analytics",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()

# Convert customer data into DataFrame
customer_df = pd.DataFrame(customer_data)

# Save customer analytics
customer_df.to_csv(
    "data/processed/customer_analytics.csv",
    index=False
)

print("Customer analytics saved successfully!")