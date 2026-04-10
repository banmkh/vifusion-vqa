from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset


class VLSPDataset(Dataset):
    def __init__(
        self,
        dataframe,
        transform: Optional[Callable] = None,
        columns: Sequence[str] = ("Anno ID", "Image ID", "Image Path", "Question", "Answer"),
    ) -> None:
        self.data = dataframe
        self.transform = transform
        self.columns = list(columns)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx][self.columns]
        anno_id, img_id, image_path, question, answer = row
        image = Image.open(Path(image_path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return anno_id, img_id, image, question, answer
