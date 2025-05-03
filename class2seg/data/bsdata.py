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
             img, _ = self.transform(img)

        # return (image, label, sample_id) so LightningModule can unpack x, y, _:
        return img, label, img_path.name



    # ------------------------------------------------------------------
    def _scan(self):
        fold_dir = self.root / self.fold
        dirs = sorted([d for d in fold_dir.iterdir()
                       if d.is_dir() and not d.name.startswith('.')])
        # now dirs == [Path(.../'defect'), Path(.../'no_defect')]
        class_to_idx = {d.name: i for i, d in enumerate(dirs)}
        # ==> {'defect': 0, 'no_defect': 1}
        samples = []
        for cls_name, cls_idx in class_to_idx.items():
            for img_path in (fold_dir/cls_name).iterdir():
                if img_path.suffix.lower() in self.IMG_EXTS:
                    samples.append((img_path, cls_idx))
        return samples


