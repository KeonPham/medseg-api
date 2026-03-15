"""Validate and stage new training data for retraining.

Usage:
    python scripts/add_new_data.py --images /path/to/images --masks /path/to/masks
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
INCOMING_DIR = Path("data/raw/incoming")


def validate_pair(image_path: Path, mask_path: Path) -> str | None:
    """Validate an image-mask pair.

    Returns None if valid, or an error message string.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return f"cannot read image: {image_path.name}"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return f"cannot read mask: {mask_path.name}"

    # Check dimensions match
    if img.shape[:2] != mask.shape[:2]:
        return (
            f"dimension mismatch: image {img.shape[:2]} vs mask {mask.shape[:2]} "
            f"for {image_path.name}"
        )

    # Check mask is binary (only 0 and 255, with tolerance for JPEG artifacts)
    unique = np.unique(mask)
    non_binary = ~np.isin(unique, [0, 255])
    if non_binary.any():
        # Allow near-binary masks (JPEG compression artifacts)
        mid_values = mask[(mask > 10) & (mask < 245)]
        if len(mid_values) > 0.01 * mask.size:
            return f"mask not binary ({len(unique)} unique values) for {image_path.name}"

    return None


def find_mask(image_path: Path, mask_dir: Path) -> Path | None:
    """Find the matching mask file for an image."""
    # Exact name match
    exact = mask_dir / image_path.name
    if exact.exists():
        return exact

    # Try matching stem with different extensions
    for ext in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{image_path.stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Add new training data")
    parser.add_argument("--images", type=str, required=True, help="Directory of new images")
    parser.add_argument("--masks", type=str, required=True, help="Directory of new masks")
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)

    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)
    if not mask_dir.is_dir():
        logger.error("Mask directory not found: %s", mask_dir)
        sys.exit(1)

    # Collect image files
    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not image_files:
        logger.error("No image files found in %s", image_dir)
        sys.exit(1)

    # Create incoming directories
    incoming_images = INCOMING_DIR / "images"
    incoming_masks = INCOMING_DIR / "masks"
    incoming_images.mkdir(parents=True, exist_ok=True)
    incoming_masks.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    errors: list[str] = []

    for img_path in image_files:
        mask_path = find_mask(img_path, mask_dir)
        if mask_path is None:
            errors.append(f"no mask found for {img_path.name}")
            rejected += 1
            continue

        error = validate_pair(img_path, mask_path)
        if error is not None:
            errors.append(error)
            rejected += 1
            continue

        # Copy validated pair
        shutil.copy2(img_path, incoming_images / img_path.name)
        shutil.copy2(mask_path, incoming_masks / img_path.name)
        accepted += 1

    # Summary
    print(f"\n{'=' * 50}")
    print("  Data Staging Summary")
    print(f"{'=' * 50}")
    print(f"  Source:    {image_dir}")
    print(f"  Dest:      {INCOMING_DIR}")
    print(f"  Accepted:  {accepted}")
    print(f"  Rejected:  {rejected}")

    if errors:
        print("\n  Errors:")
        for e in errors:
            print(f"    - {e}")

    if accepted > 0:
        print(f"\n  {accepted} image-mask pairs staged in {INCOMING_DIR}")
    else:
        print("\n  No valid pairs found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
