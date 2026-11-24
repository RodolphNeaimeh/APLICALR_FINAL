from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]

folder = ROOT / "data" / "01_raw" / "20261116" / "Vidéos annotées"

for video in sorted(folder.glob("*.mp4")):

    print(f"\nVideo: {video.name}")

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print("ERROR: Cannot open video")
        cap.release()
        continue

    ret, frame = cap.read()

    if ret:
        print("Status: OK")
        print("FPS:", cap.get(cv2.CAP_PROP_FPS))
        print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        print("Resolution:", frame.shape)
    else:
        print("ERROR: Cannot read frame")

    cap.release()