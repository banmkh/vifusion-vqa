from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data import convert_json_to_csv


def test_convert_json_to_csv(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "1.jpg"
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(img_path)

    data = {
        "images": {"1": "1.jpg"},
        "annotations": {
            "a1": {"image_id": 1, "question": "Q1", "answer": "A1"},
            "a2": {"image_id": 1, "question": "Q2", "answer": "A2"},
        },
    }

    json_path = tmp_path / "ann.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")

    csv_path = tmp_path / "out.csv"
    convert_json_to_csv(json_path, images_dir, csv_path, limit=1)

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["Question"] == "Q1"
    assert df.iloc[0]["Answer"] == "A1"
    assert Path(df.iloc[0]["Image Path"]).name == "1.jpg"
