import cv2
from ultralytics import YOLO
import pandas as pd

# ==============================
# Load YOLO model
# ==============================

model = YOLO("yolov8n.pt")

# ==============================
# Input video
# ==============================

video_path = "video/retail_store.mp4"

cap = cv2.VideoCapture(video_path)

# Check video
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# ==============================
# Video dimensions
# ==============================

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Virtual counting line
line_y = frame_height // 2

# ==============================
# Variables
# ==============================

previous_positions = {}

entered_ids = set()
exited_ids = set()

analytics_data = []
frame_number = 0

# Occupancy statistics
occupancy_history = []
peak_occupancy = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # ==============================
    # YOLO Tracking
    # ==============================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    annotated_frame = results[0].plot()

    current_ids = set()

    # ==============================
    # Process detected people
    # ==============================

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        for box, track_id in zip(boxes, boxes.id):

            person_id = int(track_id)

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Calculate center of person
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            current_ids.add(person_id)

            # ==============================
            # Check previous position
            # ==============================

            if person_id in previous_positions:

                previous_y = previous_positions[person_id]

                # Person moving DOWN across line
                if previous_y < line_y and center_y >= line_y:

                    if person_id not in entered_ids:
                        entered_ids.add(person_id)

                # Person moving UP across line
                elif previous_y > line_y and center_y <= line_y:

                    if person_id not in exited_ids:
                        exited_ids.add(person_id)

            # Save current position
            previous_positions[person_id] = center_y

            # Display customer ID
            cv2.putText(
                annotated_frame,
                f"ID: {person_id}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    # ==============================
    # Current customers
    # ==============================

    current_customers = len(current_ids)
    # Store occupancy history
    occupancy_history.append(current_customers)

    # Update peak occupancy
    if current_customers > peak_occupancy:
        peak_occupancy = current_customers
    # ==============================
    # Draw virtual line
    # ==============================

    cv2.line(
        annotated_frame,
        (0, line_y),
        (frame_width, line_y),
        (0, 0, 255),
        3
    )

    cv2.putText(
        annotated_frame,
        "COUNTING LINE",
        (20, line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # ==============================
    # Display analytics
    # ==============================

    cv2.putText(
        annotated_frame,
        f"Current Customers: {current_customers}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Peak Occupancy: {peak_occupancy}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
        

    cv2.putText(
        annotated_frame,
        f"Entered: {len(entered_ids)}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Exited: {len(exited_ids)}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    # ==============================
    # Save frame analytics
    # ==============================

    analytics_data.append({
        "Frame": frame_number,
        "Current_Customers": current_customers,
        "Total_Entered": len(entered_ids),
        "Total_Exited": len(exited_ids)
    })

    # ==============================
    # Display video
    # ==============================

    cv2.imshow(
        "Retail Customer Entry Exit Analytics",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# Release resources
# ==============================

cap.release()
cv2.destroyAllWindows()


# ==============================
# Save analytics
# ==============================

customer_df = pd.DataFrame(analytics_data)

customer_df.to_csv(
    "data/processed/customer_entry_exit.csv",
    index=False
)

print("\n================================")
print("Customer Entry/Exit Analytics")
print("================================")

print("Total Customers Entered:", len(entered_ids))
print("Total Customers Exited:", len(exited_ids))
print("Total Frames Analyzed:", frame_number)
average_occupancy = sum(occupancy_history) / len(occupancy_history)

print(
    "Average Occupancy:",
    round(average_occupancy, 2)
)

print(
    "Peak Occupancy:",
    peak_occupancy
)
print("\nAnalytics saved successfully!")