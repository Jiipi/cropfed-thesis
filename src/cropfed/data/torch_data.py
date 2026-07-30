"""PyTorch datasets backed by CSV manifests, keeping images at each client."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cropfed.data.manifest import read_manifest


def build_transforms(*, training: bool):
    """Return ImageNet-compatible transforms for transfer learning."""

    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ManifestImageDataset:
    """Dataset wrapper with lazy PyTorch inheritance requirements.

    The object implements the Dataset protocol and therefore works with
    ``torch.utils.data.DataLoader`` without importing torch at module load time.
    """

    def __init__(self, manifest_path: Path, *, training: bool) -> None:
        self.records = read_manifest(manifest_path)
        if not self.records:
            raise ValueError(f"manifest is empty: {manifest_path}")
        self.transform = build_transforms(training=training)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        return self.transform(image), record.label_id


def build_dataloader(
    manifest_path: Path,
    *,
    training: bool,
    batch_size: int,
    num_workers: int = 0,
):
    import torch
    from torch.utils.data import DataLoader

    dataset = ManifestImageDataset(manifest_path, training=training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
