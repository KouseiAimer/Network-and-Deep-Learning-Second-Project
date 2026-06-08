"""
Data loading helpers for CIFAR-10 Batch Normalization experiments.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class PartialDataset(Dataset):
    """A deterministic prefix subset for quick debug runs."""

    def __init__(self, dataset: Dataset, n_items: int) -> None:
        self.dataset = dataset
        self.n_items = int(n_items)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return min(self.n_items, len(self.dataset))


def _build_transform(train: bool, augment: bool) -> transforms.Compose:
    transform_list = []
    if train and augment:
        transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(transform_list)


def get_cifar_loader(
    root: str | Path,
    batch_size: int = 128,
    train: bool = True,
    shuffle: bool | None = None,
    num_workers: int = 0,
    n_items: int = -1,
    augment: bool = False,
) -> DataLoader:
    if shuffle is None:
        shuffle = train

    dataset = datasets.CIFAR10(
        root=str(root),
        train=train,
        download=True,
        transform=_build_transform(train=train, augment=augment),
    )
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_cifar_loaders(
    root: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    n_train_items: int = -1,
    n_val_items: int = -1,
    augment: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_loader = get_cifar_loader(
        root=root,
        batch_size=batch_size,
        train=True,
        shuffle=True,
        num_workers=num_workers,
        n_items=n_train_items,
        augment=augment,
    )
    val_loader = get_cifar_loader(
        root=root,
        batch_size=batch_size,
        train=False,
        shuffle=False,
        num_workers=num_workers,
        n_items=n_val_items,
        augment=False,
    )
    return train_loader, val_loader
