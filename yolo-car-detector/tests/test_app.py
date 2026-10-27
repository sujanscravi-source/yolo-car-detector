from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


def test_uses_local_release_weights_when_present(tmp_path, monkeypatch):
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"demo weights")
    monkeypatch.setattr(app, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(app, "MODEL_NAME", "yolo11n.pt")

    assert app.model_source() == str(weights)


def test_falls_back_to_ultralytics_model_download_name(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(app, "MODEL_NAME", "yolov8n.pt")

    assert app.model_source() == "yolov8n.pt"


def test_detect_rejects_missing_image_before_loading_yolo(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image not found"):
        app.detect_cars(tmp_path / "missing.jpg")


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_detect_rejects_invalid_confidence_before_loading_yolo(tmp_path, confidence):
    image = tmp_path / "car.jpg"
    image.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="between 0 and 1"):
        app.detect_cars(image, confidence)
