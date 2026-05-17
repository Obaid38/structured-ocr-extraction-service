#!/usr/bin/env python3
"""
Generic YOLO Inference Script
==============================
Run any YOLO model on any file/folder and output annotated PNGs
with bounding boxes, class names, and confidence scores.

Usage:
    # Single image
    python generic_inference.py --model path/to/best.pt --input image.jpg

    # Single PDF
    python generic_inference.py --model path/to/best.pt --input doc.pdf

    # Folder of images/PDFs
    python generic_inference.py --model path/to/best.pt --input ./test_folder/

    # Custom confidence & output dir
    python generic_inference.py --model path/to/best.pt --input image.jpg --conf 0.5 --output ./results/
"""

import argparse
import colorsys
import sys
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".pdf"}


def generate_colors(n: int):
    """Generate n visually distinct BGR colors."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


def pdf_to_images(pdf_path: Path, dpi: int = 200):
    """Convert PDF pages to list of BGR numpy arrays."""
    import fitz  # PyMuPDF

    images = []
    doc = fitz.open(str(pdf_path))
    for page in doc:
        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images


def draw_boxes(image: np.ndarray, boxes, class_names: dict, colors: list) -> np.ndarray:
    """Draw bounding boxes with class name and confidence on the image."""
    img = image.copy()
    h, w = img.shape[:2]
    line_width = max(2, int(min(h, w) * 0.002))
    font_scale = max(0.4, min(h, w) / 1800)
    font_thickness = max(1, line_width // 2)

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = colors[cls_id % len(colors)]
        label = f"{class_names.get(cls_id, cls_id)} {conf:.2f}"

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_y1 = max(y1 - th - 10, 0)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 6, label_y1 + th + 10), color, -1)

        # Label text
        cv2.putText(img, label, (x1 + 3, label_y1 + th + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return img


def run_inference(model_path: str, input_path: str, output_dir: str, conf: float, iou: float, dpi: int):
    from ultralytics import YOLO

    model = YOLO(model_path)
    class_names = model.names  # dict {0: 'class_a', 1: 'class_b', ...}
    colors = generate_colors(len(class_names))

    print(f"Model:   {model_path}")
    print(f"Classes: {class_names}")
    print(f"Conf:    {conf}  |  IoU: {iou}")
    print()

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    if input_path.is_dir():
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
    elif input_path.is_file():
        files = [input_path]
    else:
        print(f"[ERROR] Path not found: {input_path}")
        sys.exit(1)

    if not files:
        print("[ERROR] No supported files found.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to process.\n")

    total_detections = 0

    for file_path in files:
        print(f"--- {file_path.name} ---")

        # Load image(s)
        if file_path.suffix.lower() == ".pdf":
            pages = pdf_to_images(file_path, dpi=dpi)
        else:
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"  [WARN] Could not read {file_path.name}, skipping.")
                continue
            pages = [img]

        for page_idx, image in enumerate(pages):
            results = model.predict(image, conf=conf, iou=iou, verbose=False)
            result = results[0]
            n_det = len(result.boxes) if result.boxes is not None else 0
            total_detections += n_det

            # Print detections
            if n_det:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    c = float(box.conf[0])
                    print(f"  Page {page_idx + 1}: {class_names[cls_id]} ({c:.3f})")
            else:
                print(f"  Page {page_idx + 1}: No detections")

            # Draw & save
            annotated = draw_boxes(image, result.boxes, class_names, colors)
            if len(pages) > 1:
                out_name = f"{file_path.stem}_page{page_idx + 1:02d}.png"
            else:
                out_name = f"{file_path.stem}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), annotated)
            print(f"  -> {out_path}")

        print()

    print(f"Done. Total detections: {total_detections}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generic YOLO inference - outputs annotated PNGs with bbox + confidence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--input", "-i", required=True, help="Input file or folder")
    parser.add_argument("--output", "-o", default="./inference_output", help="Output directory for annotated PNGs")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering")

    args = parser.parse_args()
    run_inference(args.model, args.input, args.output, args.conf, args.iou, args.dpi)


if __name__ == "__main__":
    main()
