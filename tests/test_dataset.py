from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.data import VLSPDataset


def test_vlsp_dataset_getitem(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "1.jpg"
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(img_path)

    df = pd.DataFrame(
        [
            {
                "Anno ID": "a1",
                "Image ID": 1,
                "Image Path": str(img_path),
                "Question": "Q?",
                "Answer": "A",
            }
        ]
    )

    ds = VLSPDataset(df)
    anno_id, img_id, image, question, answer = ds[0]

    assert anno_id == "a1"
    assert img_id == 1
    assert question == "Q?"
    assert answer == "A"
    assert image.size == (8, 8)
