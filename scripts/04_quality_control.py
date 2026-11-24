from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

IMAGES_DIR = ROOT / "data/03_extracted/original_frames"
MASKS_DIR = ROOT / "data/05_annotations/masks"
OUTPUT_DIR = ROOT / "data/02_inventory"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

for image_path in sorted(IMAGES_DIR.glob("*.png")):

    mask_path = MASKS_DIR / image_path.name

    issues = []

    if not mask_path.exists():
        records.append({
            "filename": image_path.name,
            "status": "ERROR",
            "issues": "Missing mask"
        })
        continue

    image = cv2.imread(str(image_path))

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None or mask is None:
        records.append({
            "filename": image_path.name,
            "status": "ERROR",
            "issues": "Unreadable file"
        })
        continue

    # 1. Check dimensions
    if image.shape[:2] != mask.shape:
        issues.append("Dimension mismatch")

    # 2. Check pixel values
    values = np.unique(mask)

    if not np.all(np.isin(values, [0, 255])):
        issues.append("Non-binary mask")

    # 3. Measure mask area
    mask_pixels = np.count_nonzero(mask)

    mask_fraction = mask_pixels / mask.size

    if mask_pixels == 0:
        issues.append("Empty mask")

    if mask_fraction < 0.001:
        issues.append("Very small mask")

    if mask_fraction > 0.15:
        issues.append("Very large mask")

    # 4. Connected components
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    # Exclude background
    num_regions = num_labels - 1

    areas = stats[1:, cv2.CC_STAT_AREA]

    small_regions = int(np.sum(areas < 50))

    if small_regions > 0:
        issues.append("Small isolated regions")

    records.append({
        "filename": image_path.name,
        "video": image_path.name.split("_")[0],
        "height": mask.shape[0],
        "width": mask.shape[1],
        "mask_pixels": mask_pixels,
        "mask_percentage": mask_fraction * 100,
        "num_regions": num_regions,
        "small_regions": small_regions,
        "status": "REVIEW" if issues else "OK",
        "issues": "; ".join(issues)
    })

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_DIR / "quality_control_report.csv",
    index=False
)

print("\n===== QUALITY CONTROL =====")

print("Total images:", len(df))

print("\nStatus distribution:")
print(df["status"].value_counts())

print("\n===== SUSPICIOUS MASKS =====")

suspect = df[df["status"] != "OK"]

print(
    suspect[["filename", "issues"]]
    .to_string(index=False)
)

print("\nReport saved successfully.")