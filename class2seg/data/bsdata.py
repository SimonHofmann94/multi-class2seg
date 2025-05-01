from __future__ import annotations
import torch
import pathlib
from typing import Callable, List, Tuple
from PIL import Image
from torch.utils.data import Dataset

__all__ = ["BSDataClassificationDataset"]


class BSDataClassificationDataset(Dataset):
    """Custom dataset for binary crack classification."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(
        self,
        root: str | pathlib.Path,
        *,
        fold: str = "train",
        transform: Callable | None = None,
        segmentation: bool = False,
        random_state: int | None = None,
        **_: object,
    ) -> None:
        if segmentation:
            raise ValueError("BSDataClassificationDataset unterstützt keine Segmentations‑Masken (segmentation=False setzen)")

        self.root = pathlib.Path(root).expanduser().resolve()
        if fold not in {"train", "val", "test"}:
            raise ValueError(f"fold must be 'train', 'val' or 'test', got {fold!r}")
        self.fold = fold
        self.transform = transform
        self.random_state = random_state

        self.samples: List[Tuple[pathlib.Path, int]] = self._scan()

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img, _ = self.transform(img)      # nur Bild übernehmen

        dummy_mask = torch.tensor(0)          # <-- Platzhalter
        return img, label, dummy_mask         #      ^ 3-tes Element




    # ------------------------------------------------------------------
    def _scan(self) -> List[Tuple[pathlib.Path, int]]:
        """Collect (image_path, class_idx) pairs from the given `fold`."""
        fold_dir = self.root / self.fold
        if not fold_dir.exists():
            raise FileNotFoundError(f"Directory {fold_dir} does not exist")

        class_to_idx = {d.name: i for i, d in enumerate(sorted(fold_dir.iterdir())) if d.is_dir()}
        if not class_to_idx:
            raise RuntimeError(f"No class sub‑directories found in {fold_dir}")

        samples: List[Tuple[pathlib.Path, int]] = []
        for class_name, class_idx in class_to_idx.items():
            for p in (fold_dir / class_name).iterdir():
                if p.suffix.lower() in self.IMG_EXTS:
                    samples.append((p, class_idx))

        if not samples:
            raise RuntimeError(f"Found zero images in {fold_dir}")

        return samples
