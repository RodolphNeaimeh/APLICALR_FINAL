from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = (
    ROOT / "data" / "01_raw" / "20261116"
    / "Vidéos annotées" / "Image004 annotée.mp4"
)

# 1. Read annotated frame
cap = cv2.VideoCapture(str(VIDEO_PATH))

cap.set(cv2.CAP_PROP_POS_FRAMES, 155)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Cannot read frame")

# 2. Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 3. Detect green pixels
lower_green = np.array([35, 70, 35])
upper_green = np.array([85, 255, 255])

green_mask = cv2.inRange(
    hsv, lower_green, upper_green
)

# 4. Close small gaps in contours
kernel = np.ones((3, 3), np.uint8)

closed_mask = cv2.morphologyEx(
    green_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# 5. Find external contours
contours, _ = cv2.findContours(
    closed_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 6. Create an empty segmentation mask
final_mask = np.zeros(
    frame.shape[:2],
    dtype=np.uint8
)

# 7. Fill detected contours
count = 0

for contour in contours:

    if cv2.contourArea(contour) > 30:

        cv2.drawContours(
            final_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        count += 1

print("Detected contours:", count)
print("Mask shape:", final_mask.shape)

# Remove thin unwanted lines
opening_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

final_mask = cv2.morphologyEx(
    final_mask,
    cv2.MORPH_OPEN,
    opening_kernel
)
# 8. Visualize the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
axes[0].set_title("Annotated image")

axes[1].imshow(green_mask, cmap="gray")
axes[1].set_title("Detected green pixels")

axes[2].imshow(final_mask, cmap="gray")
axes[2].set_title("Generated segmentation mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()