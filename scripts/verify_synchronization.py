from pathlib import Path
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "01_raw" / "20261116"

original_path = BASE / "Image004.mp4"

annotated_path = (
    BASE / "Vidéos annotées" / "Image004 annotée.mp4"
)


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {index}")

    return frame


def compare_frames(original, annotated):

    if original.shape != annotated.shape:
        raise RuntimeError("Different image dimensions")

    # Convert images to grayscale
    gray_original = cv2.cvtColor(
        original, cv2.COLOR_BGR2GRAY
    )

    gray_annotated = cv2.cvtColor(
        annotated, cv2.COLOR_BGR2GRAY
    )

    # Detect green annotations
    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 70, 35]),
        np.array([85, 255, 255])
    )

    # Exclude annotation pixels and their surroundings
    kernel = np.ones((11, 11), np.uint8)

    green_mask = cv2.dilate(green_mask, kernel)

    # Compute structural similarity map
    _, similarity_map = ssim(
        gray_original,
        gray_annotated,
        data_range=255,
        full=True
    )

    # Exclude black background and annotations
    valid_pixels = (
        (green_mask == 0) &
        (gray_original > 20) &
        (gray_annotated > 20)
    )

    if not np.any(valid_pixels):
        raise RuntimeError("No valid comparison pixels")

    return similarity_map[valid_pixels].mean()


# Test different moments of the video
test_indices = [30, 75, 155, 220, 280]

for index in test_indices:

    annotated = read_frame(annotated_path, index)

    scores = []

    print(f"\n===== Annotated frame {index} =====")

    for offset in range(-5, 6):

        candidate_index = index + offset

        original = read_frame(
            original_path, candidate_index
        )

        score = compare_frames(original, annotated)

        scores.append((candidate_index, score))

        print(
            f"Original frame {candidate_index}: "
            f"SSIM = {score:.5f}"
        )

    best_index, best_score = max(
        scores, key=lambda x: x[1]
    )

    print(f"Best matching frame: {best_index}")
    print(f"Best SSIM: {best_score:.5f}")
    print(f"Offset: {best_index - index}")