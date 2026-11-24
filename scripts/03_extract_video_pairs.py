from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# ==========================
# 1. CONFIGURATION
# ==========================

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "data" / "01_raw" / "20261116"
ANNOTATED_DIR = BASE / "Vidéos annotées"

REPORT_PATH = (
    ROOT / "data" / "02_inventory"
    / "synchronization_report.csv"
)

IMAGES_DIR = (
    ROOT / "data" / "03_extracted"
    / "original_frames"
)

MASKS_DIR = (
    ROOT / "data" / "05_annotations" / "masks"
)

OVERLAYS_DIR = (
    ROOT / "data" / "05_annotations" / "overlays"
)

for folder in [IMAGES_DIR, MASKS_DIR, OVERLAYS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Treat the following video IDs
VIDEO_IDS = [1, 4, 6, 8, 10, 12]

SSIM_THRESHOLD = 0.98

# Extract one frame out of every five
FRAME_STEP = 5

report = pd.read_csv(REPORT_PATH)

records = []


# ==========================
# 2. MASK GENERATION
# ==========================

def generate_mask(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 70, 35])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(
        hsv, lower_green, upper_green
    )

    # Close small gaps
    closing_kernel = np.ones((3, 3), np.uint8)

    closed_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        closing_kernel
    )

    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    for contour in contours:

        if cv2.contourArea(contour) > 30:

            cv2.drawContours(
                final_mask,
                [contour],
                -1,
                255,
                cv2.FILLED
            )

    # Remove thin unwanted lines
    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_OPEN,
        opening_kernel
    )

    return final_mask


# ==========================
# 3. OVERLAY GENERATION
# ==========================

def create_overlay(image, mask):

    overlay = image.copy()

    overlay[mask > 0] = (0, 255, 0)

    return cv2.addWeighted(
        image, 0.65,
        overlay, 0.35,
        0
    )


# ==========================
# 4. VIDEO PROCESSING
# ==========================

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

    cap_o = cv2.VideoCapture(str(original_path))
    cap_a = cv2.VideoCapture(str(annotated_path))

    if not cap_o.isOpened() or not cap_a.isOpened():
        print(f"Cannot open {name}")
        cap_o.release()
        cap_a.release()
        continue

    video_report = report[
        report["video"] == name
    ]

    scores = dict(zip(
        video_report["frame"],
        video_report["ssim"]
    ))

    index = 0
    saved = 0
    rejected = 0

    print(f"\nProcessing {name}")

    while True:

        ret_o, original = cap_o.read()
        ret_a, annotated = cap_a.read()

        if not ret_o or not ret_a:
            break

        if index % FRAME_STEP != 0:
            index += 1
            continue

        score = scores.get(index, 0)

        if (
            pd.isna(score)
            or score < SSIM_THRESHOLD
            or original.shape != annotated.shape
        ):
            rejected += 1
            index += 1
            continue

        mask = generate_mask(annotated)

        # Do not assume an empty annotation
        # means that no anatomical structure exists
        if np.count_nonzero(mask) == 0:
            rejected += 1
            index += 1
            continue

        filename = f"{name}_frame_{index:04d}.png"

        overlay = create_overlay(original, mask)

        # Save original image
        ok_image = cv2.imwrite(
            str(IMAGES_DIR / filename),
            original
        )

        # Save segmentation mask
        ok_mask = cv2.imwrite(
            str(MASKS_DIR / filename),
            mask
        )

        # Save overlay
        ok_overlay = cv2.imwrite(
            str(OVERLAYS_DIR / filename),
            overlay
        )

        if not all([ok_image, ok_mask, ok_overlay]):
            raise RuntimeError(
                f"Cannot save {filename}"
            )

        records.append({
            "video": name,
            "frame": index,
            "filename": filename,
            "ssim": score,
            "mask_pixels": np.count_nonzero(mask),
            "mask_fraction": np.count_nonzero(mask) / mask.size,
            "quality_status": "pending_review"
        })

        saved += 1
        index += 1

    cap_o.release()
    cap_a.release()

    print("Saved pairs:", saved)
    print("Rejected sampled frames:", rejected)


# ==========================
# 5. SAVE EXTRACTION REPORT
# ==========================

output_report = (
    ROOT / "data" / "02_inventory"
    / "extraction_report.csv"
)

pd.DataFrame(records).to_csv(
    output_report,
    index=False
)

print("\n===== EXTRACTION COMPLETE =====")
print("Total saved pairs:", len(records))