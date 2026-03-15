"""Call the MedSegAPI and save annotated results to the results/ folder.

Usage:
    python scripts/predict_and_save.py IMAGE_PATH [--api-key KEY] [--model MODEL] [--host HOST]
    python scripts/predict_and_save.py /path/to/xray.png --api-key YOUR_KEY
    python scripts/predict_and_save.py /path/to/folder/ --api-key YOUR_KEY  # batch all PNGs
"""

import argparse
import base64
import io
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode a base64 PNG string to a numpy RGB array."""
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img.convert("RGB"))


def draw_info_panel(
    original: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    filename: str,
    model_name: str,
    model_version: str,
    inference_time_ms: float,
    metrics: dict,
    image_size: dict,
) -> np.ndarray:
    """Compose a single annotated output image with original, mask, overlay, and metrics."""
    h, w = original.shape[:2]

    # Resize all to same height
    target_h = max(h, 400)
    scale = target_h / h

    def resize(img: np.ndarray) -> np.ndarray:
        new_w = int(img.shape[1] * scale)
        new_h = int(img.shape[0] * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    orig_r = resize(original)
    overlay_r = resize(overlay)

    # Convert mask to 3-channel for display
    if mask.ndim == 2:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        mask_3ch = mask
    mask_r = resize(mask_3ch)

    panel_h = orig_r.shape[0]
    panel_w = orig_r.shape[1]

    # Info panel width
    info_w = 320
    gap = 8

    # Total canvas
    total_w = panel_w * 3 + gap * 2 + info_w + gap
    canvas = np.zeros((panel_h + 80, total_w, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # dark gray background

    # Place images
    y_off = 50
    x = gap
    canvas[y_off : y_off + panel_h, x : x + panel_w] = orig_r
    x += panel_w + gap
    canvas[y_off : y_off + panel_h, x : x + panel_w] = mask_r
    x += panel_w + gap
    canvas[y_off : y_off + panel_h, x : x + panel_w] = overlay_r

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil_img)

    # Try to use a monospace font
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_title = ImageFont.load_default()

    white = (255, 255, 255)
    green = (0, 230, 118)
    cyan = (0, 200, 255)
    yellow = (255, 235, 59)
    gray = (180, 180, 180)

    # Column labels
    col_x = gap + panel_w // 2
    draw.text((col_x - 30, 15), "Original", fill=white, font=font_large, anchor="mm")
    col_x = gap + panel_w + gap + panel_w // 2
    draw.text((col_x - 30, 15), "Mask", fill=white, font=font_large, anchor="mm")
    col_x = gap + (panel_w + gap) * 2 + panel_w // 2
    draw.text((col_x - 30, 15), "Overlay", fill=white, font=font_large, anchor="mm")

    # Info panel on the right
    info_x = gap + (panel_w + gap) * 3
    iy = y_off + 5

    draw.text((info_x, iy), "MedSegAPI Results", fill=cyan, font=font_title)
    iy += 30
    draw.line([(info_x, iy), (info_x + info_w - 20, iy)], fill=gray, width=1)
    iy += 12

    lines = [
        (f"File:  {filename}", white),
        (f"Size:  {image_size.get('width', '?')}x{image_size.get('height', '?')}", white),
        ("", white),
        (f"Model:    {model_name}", green),
        (f"Version:  {model_version}", green),
        (f"Time:     {inference_time_ms:.1f} ms", yellow),
        ("", white),
        ("--- Metrics ---", cyan),
        (f"Confidence:  {metrics.get('confidence_score', 0) * 100:.1f}%", white),
        (f"Lung Cover:  {metrics.get('lung_coverage_pct', 0):.1f}%", white),
        (f"Symmetry:    {metrics.get('symmetry_ratio', 0):.3f}", white),
    ]

    for text, color in lines:
        if text:
            draw.text((info_x, iy), text, fill=color, font=font_small)
        iy += 20

    # Symmetry assessment
    iy += 5
    sym = metrics.get("symmetry_ratio", 0)
    if sym > 0.85:
        sym_label = "Symmetric (normal)"
        sym_color = green
    elif sym > 0.6:
        sym_label = "Moderate asymmetry"
        sym_color = yellow
    else:
        sym_label = "Significant asymmetry"
        sym_color = (255, 80, 80)
    draw.text((info_x, iy), sym_label, fill=sym_color, font=font_small)

    # Bottom bar
    bottom_y = panel_h + y_off + 10
    disclaimer = "Research use only - Not for clinical diagnosis"
    draw.text((gap, bottom_y), disclaimer, fill=(120, 120, 120), font=font_small)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((total_w - 200, bottom_y), ts, fill=(120, 120, 120), font=font_small)

    return np.array(pil_img)


def predict_single(
    image_path: Path,
    api_key: str,
    host: str,
    model_name: str,
    results_dir: Path,
) -> bool:
    """Send one image to the API and save annotated results."""
    print(f"  Processing: {image_path.name}")

    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{host}/api/v1/predict?model_name={model_name}&return_overlay=true",
            headers={"X-API-Key": api_key} if api_key else {},
            files={"file": (image_path.name, f, "image/png")},
            timeout=60,
        )

    if resp.status_code != 200:
        print(f"    ERROR {resp.status_code}: {resp.text[:200]}")
        return False

    data = resp.json()

    # Load original for composite
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Decode mask
    mask_rgb = decode_base64_image(data["mask_base64"])
    mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)

    # Decode overlay
    overlay = decode_base64_image(data["overlay_base64"])

    # Build annotated composite
    composite = draw_info_panel(
        original=original,
        mask=mask_gray,
        overlay=overlay,
        filename=image_path.name,
        model_name=data["model_name"],
        model_version=data["model_version"],
        inference_time_ms=data["inference_time_ms"],
        metrics=data["metrics"],
        image_size=data["image_size"],
    )

    # Save individual files
    stem = image_path.stem
    sub = results_dir / stem
    sub.mkdir(parents=True, exist_ok=True)

    # Save composite
    composite_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(sub / f"{stem}_result.png"), composite_bgr)

    # Save individual outputs
    cv2.imwrite(str(sub / f"{stem}_mask.png"), mask_gray)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(sub / f"{stem}_overlay.png"), overlay_bgr)

    metrics = data["metrics"]
    print(f"    Model: {data['model_name']} {data['model_version']}")
    print(f"    Time:  {data['inference_time_ms']:.1f} ms")
    print(f"    Conf:  {metrics['confidence_score'] * 100:.1f}%")
    print(f"    Cover: {metrics['lung_coverage_pct']:.1f}%")
    print(f"    Saved: {sub}/")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedSegAPI prediction and save annotated results")
    parser.add_argument("path", help="Image file or directory of images")
    parser.add_argument("--api-key", default="", help="API key (omit if auth disabled)")
    parser.add_argument("--model", default="hybrid", help="Model: hybrid, cnn_only, vit_only")
    parser.add_argument("--host", default="http://localhost:8000", help="API host")
    parser.add_argument("--output", default="results", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.path)
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect images
    if input_path.is_dir():
        images = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
        if not images:
            print(f"No PNG/JPG images found in {input_path}")
            sys.exit(1)
        print(f"Found {len(images)} images in {input_path}")
    elif input_path.is_file():
        images = [input_path]
    else:
        print(f"Path not found: {input_path}")
        sys.exit(1)

    print(f"Model: {args.model} | Output: {results_dir}/\n")

    success = 0
    for img_path in images:
        if predict_single(img_path, args.api_key, args.host, args.model, results_dir):
            success += 1

    print(f"\nDone: {success}/{len(images)} predictions saved to {results_dir}/")


if __name__ == "__main__":
    main()
