"""Run a YOLO nano detector on one image and save COCO car detections."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


# AutoHeal updates this value to the filename attached to a GitHub release.
MODEL_NAME = 'yolo11n.pt'
MODEL_DIR = Path(__file__).resolve().parent / "models"
CAR_CLASS_ID = 2  # COCO class index for "car".


def model_source() -> str:
    """Use the release-downloaded weights when present, else Ultralytics' model name."""
    local_weights = MODEL_DIR / MODEL_NAME
    return str(local_weights) if local_weights.is_file() else MODEL_NAME


def detect_cars(image_path: str | Path, confidence: float = 0.25) -> list[Any]:
    """Detect cars in an image and save an annotated copy under runs/detect/cars."""
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    # Import lazily so the AutoHeal unit tests do not need PyTorch installed.
    from ultralytics import YOLO

    detector = YOLO(model_source())
    return detector.predict(
        source=str(image),
        conf=confidence,
        classes=[CAR_CLASS_ID],
        save=True,
        project="runs/detect",
        name="cars",
        exist_ok=True,
        verbose=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect cars in an image with YOLO.")
    parser.add_argument("image", help="Path to a local image file")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    results = detect_cars(args.image, args.confidence)
    count = sum(len(result.boxes) for result in results if result.boxes is not None)
    print(f"Detected {count} car(s). Annotated image: runs/detect/cars/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
