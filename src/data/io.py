from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


CSV_COLUMNS = ["Anno ID", "Image ID", "Image Path", "Question", "Answer"]


def convert_json_to_csv(
    input_file: str | Path,
    images_folder: str | Path,
    output_file: str | Path,
    limit: int | None = None,
) -> None:
    input_file = Path(input_file)
    images_folder = Path(images_folder)
    output_file = Path(output_file)

    with input_file.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    annotations = data.get("annotations", {})
    images = data.get("images", {})

    rows: list[list[str]] = [CSV_COLUMNS]

    count = 0
    for anno_id in annotations:
        if limit is not None and count >= limit:
            break
        annotation = annotations[anno_id]
        image_id = annotation.get("image_id")
        question = annotation.get("question", "")
        answer = annotation.get("answer", "")
        image_name = images.get(str(image_id), "")
        image_path = str(images_folder / image_name)
        rows.append([str(anno_id), str(image_id), image_path, question, str(answer)])
        count += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_csv_rows(output_file: str | Path, rows: Iterable[Iterable[str]]) -> None:
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
