from pathlib import Path
import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "01_raw" / "20261116"

original_path = BASE / "Image004.mp4"

annotated_path = (
    BASE / "Vidéos annotées" / "Image004 annotée.mp4"
)

def read_frame(path, index):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path.name}")

    print(f"\nVideo: {path.name}")
    print("FPS:", cap.get(cv2.CAP_PROP_FPS))
    print("Frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("Height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Cannot read frame")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


frame_index = 155

original = read_frame(original_path, frame_index)
annotated = read_frame(annotated_path, frame_index)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

axes[0].imshow(original)
axes[0].set_title("Original - Frame 155")

axes[1].imshow(annotated)
axes[1].set_title("Annotated - Frame 155")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()