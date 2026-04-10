from __future__ import annotations

from torch.utils.data import DataLoader

from .dataset import VLSPDataset


def build_dataloader(
    dataframe,
    transform,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
):
    dataset = VLSPDataset(dataframe, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_dataloaders(
    df_train,
    df_test,
    df_dev,
    transform,
    batch_size: int,
    num_workers: int = 0,
    shuffle_train: bool = True,
):
    train_loader = build_dataloader(
        df_train, transform, batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    test_loader = build_dataloader(
        df_test, transform, batch_size, shuffle=False, num_workers=num_workers
    )
    dev_loader = build_dataloader(
        df_dev, transform, batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader, dev_loader
