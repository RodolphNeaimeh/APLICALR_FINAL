from pathlib import Path
import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

video_path = (
    ROOT / "data" / "01_raw" / "20261116"
    / "Vidéos annotées" / "Image004 annotée.mp4"
)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Select the middle frame
frame_index = total_frames // 2

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ret, frame = cap.read()

cap.release()

if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 8))
    plt.imshow(frame_rgb)
    plt.title(f"Annotated video - Frame {frame_index}")
    plt.axis("off")
    plt.show()

else:
    print("Error reading frame")