import cv2
from ultralytics import YOLO
import pandas as pd

print("================================")
print("Customer Dwell Time Analytics")
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

# Get video FPS BEFORE processing
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

print(f"Video FPS: {fps:.2f}")

# =================================
# Customer Tracking Data
# =================================

entry_frames = {}
last_seen_frames = {}

frame_number = 0

# Number of frames a customer can disappear
# before we consider them to have left
max_missing_frames = int(fps * 2)

# =================================
# Process Video
# =================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Track only people
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    annotated_frame = results[0].plot()

    current_ids = set()

    # =================================
    # Process Detected Customers
    # =================================

    if results[0].boxes.id is not None:

        for box, track_id in zip(
            results[0].boxes,
            results[0].boxes.id
        ):

            customer_id = int(track_id)

            current_ids.add(customer_id)

            # First frame customer was detected
            if customer_id not in entry_frames:

                entry_frames[customer_id] = frame_number

            # Update last seen frame
            last_seen_frames[customer_id] = frame_number

            # Calculate current dwell time
            dwell_frames = (
                frame_number
                - entry_frames[customer_id]
            )

            dwell_seconds = dwell_frames / fps

            # Get bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Display ID and dwell time
            cv2.putText(
                annotated_frame,
                f"ID: {customer_id} | "
                f"Dwell: {dwell_seconds:.1f}s",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # =================================
    # Display Video
    # =================================

    cv2.putText(
        annotated_frame,
        f"Customers: {len(current_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Customer Dwell Time Analytics",
        annotated_frame
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
# Calculate Dwell Time
# =================================

dwell_data = []

for customer_id in entry_frames:

    start_frame = entry_frames[customer_id]

    end_frame = last_seen_frames[customer_id]

    dwell_frames = end_frame - start_frame

    dwell_seconds = dwell_frames / fps

    dwell_data.append({
        "Customer_ID": customer_id,
        "Entry_Frame": start_frame,
        "Exit_Frame": end_frame,
        "Dwell_Time_Seconds": round(
            dwell_seconds,
            2
        )
    })

# =================================
# Create DataFrame
# =================================

dwell_df = pd.DataFrame(dwell_data)

# =================================
# Save Results
# =================================

dwell_df.to_csv(
    "data/processed/customer_dwell_time.csv",
    index=False
)

# =================================
# Display Results
# =================================

print("\n===== Dwell Time Analytics =====")

print(
    "Customers Tracked:",
    len(dwell_df)
)

if len(dwell_df) > 0:

    print(
        "Average Dwell Time:",
        round(
            dwell_df["Dwell_Time_Seconds"].mean(),
            2
        ),
        "seconds"
    )

    print(
        "Maximum Dwell Time:",
        round(
            dwell_df["Dwell_Time_Seconds"].max(),
            2
        ),
        "seconds"
    )

    print(
        "Minimum Dwell Time:",
        round(
            dwell_df["Dwell_Time_Seconds"].min(),
            2
        ),
        "seconds"
    )

    print("\nCustomer Dwell Time Details:")

    print(dwell_df)

else:

    print("No customers were tracked.")

print(
    "\nDwell time analytics saved successfully!"
)