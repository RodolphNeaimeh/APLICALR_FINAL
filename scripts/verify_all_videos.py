from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data" / "01_raw" / "20261116"
ANNOTATED_DIR = BASE / "Vidéos annotées"

OUTPUT_DIR = ROOT / "data" / "02_inventory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_IDS = [1, 4, 6, 8, 10, 12]

records = []


def compare_frames(original, annotated):

    if original.shape != annotated.shape:
        return None

    gray_original = cv2.cvtColor(
        original, cv2.COLOR_BGR2GRAY
    )

    gray_annotated = cv2.cvtColor(
        annotated, cv2.COLOR_BGR2GRAY
    )

    hsv = cv2.cvtColor(
        annotated, cv2.COLOR_BGR2HSV
    )

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 70, 35]),
        np.array([85, 255, 255])
    )

    green_mask = cv2.dilate(
        green_mask,
        np.ones((11, 11), np.uint8)
    )

    _, similarity_map = ssim(
        gray_original,
        gray_annotated,
        data_range=255,
        full=True
    )

    valid_pixels = (
        (green_mask == 0)
        & (gray_original > 20)
        & (gray_annotated > 20)
    )

    if not np.any(valid_pixels):
        return None

    return float(similarity_map[valid_pixels].mean())


for video_id in VIDEO_IDS:

    name = f"Image{video_id:03d}"

    original_path = BASE / f"{name}.mp4"

    annotated_files = list(
        ANNOTATED_DIR.glob(f"{name}*.mp4")
    )

    if len(annotated_files) != 1:
        print(f"Missing or ambiguous video: {name}")
        continue

    annotated_path = annotated_files[0]

    cap_original = cv2.VideoCapture(
        str(original_path)
    )

    cap_annotated = cv2.VideoCapture(
        str(annotated_path)
    )

    if not cap_original.isOpened() or not cap_annotated.isOpened():

        print(f"Cannot open: {name}")

        cap_original.release()
        cap_annotated.release()

        continue

    n_original = int(
        cap_original.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    n_annotated = int(
        cap_annotated.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps_original = cap_original.get(cv2.CAP_PROP_FPS)
    fps_annotated = cap_annotated.get(cv2.CAP_PROP_FPS)

    print(f"\n===== {name} =====")
    print("Original frames:", n_original)
    print("Annotated frames:", n_annotated)
    print("Original FPS:", fps_original)
    print("Annotated FPS:", fps_annotated)

    frame_index = 0
    scores = []

    while True:

        ret_o, original = cap_original.read()
        ret_a, annotated = cap_annotated.read()

        if not ret_o or not ret_a:
            break

        score = compare_frames(
            original, annotated
        )

        if score is not None:
            scores.append(score)

        records.append({
            "video": name,
            "frame": frame_index,
            "ssim": score,
            "candidate_aligned": (
                score is not None and score >= 0.98
            )
        })

        frame_index += 1

    cap_original.release()
    cap_annotated.release()

    print("Compared frames:", frame_index)
    print("Valid scores:", len(scores))

    if scores:

        scores = np.array(scores)

        print(f"Mean SSIM: {scores.mean():.5f}")
        print(f"Minimum SSIM: {scores.min():.5f}")

        accepted = np.sum(scores >= 0.98)

        print(
            f"Frames above 0.98: "
            f"{accepted}/{frame_index}"
        )

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_DIR / "synchronization_report.csv",
    index=False
)

print("\nSynchronization report saved.")