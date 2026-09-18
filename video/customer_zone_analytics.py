import cv2
from ultralytics import YOLO
import pandas as pd

print("================================")
print("Customer Zone Analytics")
print("================================")

# =================================
# Load YOLO Model
# =================================

model = YOLO("yolov8n.pt")

# =================================
# Input Video
# =================================

video_path = "video/retail_store.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# =================================
# Zone Data
# =================================

zone_data = []

frame_number = 0

# =================================
# Process Video
# =================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Divide screen into 3 zones
    zone_width = frame_width // 3

    # =================================
    # Draw Zone Boundaries
    # =================================

    cv2.line(
        frame,
        (zone_width, 0),
        (zone_width, frame_height),
        (255, 0, 0),
        2
    )

    cv2.line(
        frame,
        (zone_width * 2, 0),
        (zone_width * 2, frame_height),
        (255, 0, 0),
        2
    )

    # Zone names
    cv2.putText(
        frame,
        "ZONE 1 - LEFT",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "ZONE 2 - CENTER",
        (zone_width + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "ZONE 3 - RIGHT",
        (zone_width * 2 + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # =================================
    # YOLO Person Tracking
    # =================================

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    # Count customers in each zone
    zone1_count = 0
    zone2_count = 0
    zone3_count = 0

    # =================================
    # Process Customers
    # =================================

    if results[0].boxes.id is not None:

        for box, track_id in zip(
            results[0].boxes,
            results[0].boxes.id
        ):

            customer_id = int(track_id)

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Find center of customer
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # =================================
            # Determine Zone
            # =================================

            if center_x < zone_width:

                zone = "Zone 1 - Left"
                zone1_count += 1

            elif center_x < zone_width * 2:

                zone = "Zone 2 - Center"
                zone2_count += 1

            else:

                zone = "Zone 3 - Right"
                zone3_count += 1

            # =================================
            # Display Customer Zone
            # =================================

            cv2.putText(
                frame,
                f"ID {customer_id}: {zone}",
                (x1, max(y1 - 10, 60)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            # Draw customer center
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 255, 0),
                -1
            )

    # =================================
    # Save Frame-Level Zone Data
    # =================================

    zone_data.append({
        "Frame": frame_number,
        "Zone_1_Left": zone1_count,
        "Zone_2_Center": zone2_count,
        "Zone_3_Right": zone3_count
    })

    # =================================
    # Display Zone Counts
    # =================================

    cv2.putText(
        frame,
        f"Left: {zone1_count}",
        (20, frame_height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Center: {zone2_count}",
        (zone_width + 20, frame_height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Right: {zone3_count}",
        (zone_width * 2 + 20, frame_height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # =================================
    # Show Video
    # =================================

    cv2.imshow(
        "Retail Store Zone Analytics",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# =================================
# Release Video
# =================================

cap.release()
cv2.destroyAllWindows()

# =================================
# Save Zone Analytics
# =================================

zone_df = pd.DataFrame(zone_data)

zone_df.to_csv(
    "data/processed/customer_zone_analytics.csv",
    index=False
)

# =================================
# Calculate Zone Statistics
# =================================

average_zone1 = zone_df[
    "Zone_1_Left"
].mean()

average_zone2 = zone_df[
    "Zone_2_Center"
].mean()

average_zone3 = zone_df[
    "Zone_3_Right"
].mean()

peak_zone1 = zone_df[
    "Zone_1_Left"
].max()

peak_zone2 = zone_df[
    "Zone_2_Center"
].max()

peak_zone3 = zone_df[
    "Zone_3_Right"
].max()

# =================================
# Display Results
# =================================

print("\n===== Zone Analytics =====")

print(
    f"Average Customers in Zone 1: "
    f"{average_zone1:.2f}"
)

print(
    f"Average Customers in Zone 2: "
    f"{average_zone2:.2f}"
)

print(
    f"Average Customers in Zone 3: "
    f"{average_zone3:.2f}"
)

print("\n===== Peak Zone Occupancy =====")

print(
    f"Zone 1 Peak: {peak_zone1}"
)

print(
    f"Zone 2 Peak: {peak_zone2}"
)

print(
    f"Zone 3 Peak: {peak_zone3}"
)

print(
    "\nZone analytics saved successfully!"
)