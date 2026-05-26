"""ADIOS-friendly PanNuke dataloader.

Reads from the unified layout produced by the cluster prepare_pannuke_data step::

    <data_dir>/<split>/<magnification>/
        tissue_images/<patch>.png
        instance_masks/<patch>.npy
        class_masks/{neoplastic,inflammatory,connective,dead,epithelial}/<patch>.png

Returns a 5-tuple per sample::

    image         FloatTensor [3, H, W]    normalized via the SynchronizedTransform
    mask_2ch      FloatTensor [2, H, W]    channel 0 = binary nuclei, channel 1 = background
    distance_map  FloatTensor [2, H, W]    channel 0 = V map, channel 1 = H map (HoVer)
    instance_mask FloatTensor [1, H, W]    per-pixel instance IDs (float, raw from .npy)
    class_mask    LongTensor  [H, W]       values in {0..5}, where 0 = background

The H/V map computation and the binary/background mask computation are the same
algorithm as ``cellvit.datasets.HoverNetBasedDataset`` (verbatim in pseudocode
from HANDOFF dataset-v2 §3). This file does not depend on ``cellvit/``.
"""

import os

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


FOREGROUND_CLASSES = ['neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial']

# Class index in the output class_mask (1-based; 0 is background).
CLASS_TO_INDEX = {name: idx + 1 for idx, name in enumerate(FOREGROUND_CLASSES)}


class ADIOSPanNukeDataset(Dataset):
    """PanNuke dataloader for the unified ADIOS layout.

    Args:
        data_dir:      root of the unified PanNuke layout (contains
                       ``Training/``, ``Test/``).
        split:         ``'Training'`` or ``'Test'`` (case-insensitive; ``'train'``
                       and ``'training'`` are accepted as ``'Training'``).
        magnification: ``'40x'`` (default) or ``'20x'``.
        transform:     a ``SynchronizedTransform`` instance (or any callable
                       that accepts ``image=..., mask=...`` and returns a
                       ``(image_tensor, mask_tensor)`` tuple). The class_mask
                       is concatenated with the instance map along the channel
                       dim before being passed through, so geometric augmentation
                       stays in sync. If ``None``, no augmentation is applied.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'Training',
        magnification: str = '40x',
        transform=None,
    ):
        self.data_dir = data_dir
        # Normalize split: 'Training' / 'Train' / 'training' all map to 'Training'.
        self.split = 'Training' if split.lower().startswith('train') else 'Test'
        self.magnification = magnification
        self.transform = transform
        self.transform_fin = A.Compose([ToTensorV2()])

        base = os.path.join(data_dir, self.split, magnification)
        self.tissue_dir = os.path.join(base, 'tissue_images')
        self.instance_dir = os.path.join(base, 'instance_masks')
        self.class_dir = os.path.join(base, 'class_masks')

        for d in (self.tissue_dir, self.instance_dir, self.class_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(f'Expected directory not found: {d}')
        for cls in FOREGROUND_CLASSES:
            cls_dir = os.path.join(self.class_dir, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f'Expected class directory not found: {cls_dir}')

        self.patch_names = self._get_patch_names()

    def _get_patch_names(self):
        names = [
            f.rsplit('.', 1)[0]
            for f in sorted(os.listdir(self.tissue_dir))
            if f.endswith('.png')
        ]
        if not names:
            raise RuntimeError(f'No .png patches found in {self.tissue_dir}')
        return names

    def __len__(self) -> int:
        return len(self.patch_names)

    # -- H/V map computation: verbatim from HoverNetBasedDataset --------------

    def _calculate_distance_maps(self, mask: np.ndarray):
        """Per-nucleus centroid-relative H/V maps.

        ``mask`` is ``[H, W, 1]``: the instance map with the trailing singleton
        dim added by ``__getitem__``. The convention matches HoverNetBasedDataset:
        instance ID values, ``max(mask)`` is the background sentinel, ``0`` is
        "nothing".

        Returns ``h_map, v_map`` each ``[H, W]`` float32, normalized to
        ``[-1, 1]`` per nucleus.
        """
        h_map = np.zeros(mask.shape[:2], dtype=np.float32)
        v_map = np.zeros(mask.shape[:2], dtype=np.float32)
        mask = np.sum(mask, axis=-1) - 1
        unique_list = np.unique(mask)
        max_value = np.max(unique_list)
        unique_list = unique_list[unique_list < max_value]
        for nucleus in unique_list:
            if nucleus <= 0:
                continue
            nucleus_mask = (mask == nucleus)
            y_indices, x_indices = np.nonzero(nucleus_mask)
            if len(y_indices) > 0:
                cy, cx = np.mean(y_indices), np.mean(x_indices)
                h_dist = x_indices - cx
                v_dist = y_indices - cy
                max_dist = max(np.max(np.abs(h_dist)), np.max(np.abs(v_dist)))
                if max_dist > 0:
                    h_dist = h_dist / max_dist
                    v_dist = v_dist / max_dist
                h_map[nucleus_mask] = h_dist
                v_map[nucleus_mask] = v_dist
        return h_map, v_map

    def _process_mask_binary(self, mask: np.ndarray) -> np.ndarray:
        """Convert instance map to binary nuclei mask (max value = background)."""
        mask = np.squeeze(mask)
        out = np.zeros(mask.shape, dtype=np.uint8)
        max_value = np.max(mask)
        out[(mask > 0) & (mask < max_value)] = 1
        return out

    # -- Class mask: stack 5 class PNGs, argmax for the foreground index -----

    def _load_class_mask(self, patch_name: str) -> np.ndarray:
        """Load per-pixel class mask for ``patch_name``.

        PanNuke labels have no pixel-level class overlap, so argmax across the
        5 class channels is a safe tiebreaker on the rare ambiguous pixel.

        Returns int64 ``[H, W]`` with values in ``{0, 1, ..., 5}``.
        """
        stacks = []
        for cls in FOREGROUND_CLASSES:
            path = os.path.join(self.class_dir, cls, f'{patch_name}.png')
            arr = np.array(Image.open(path))
            stacks.append(arr)
        stacked = np.stack(stacks, axis=0)              # [5, H, W]
        has_fg = stacked.max(axis=0) > 0                # [H, W] bool
        class_argmax = stacked.argmax(axis=0)           # [H, W] in {0..4}
        class_mask = np.where(has_fg, class_argmax + 1, 0)
        return class_mask.astype(np.int64)

    # -- __getitem__ ----------------------------------------------------------

    def __getitem__(self, idx: int):
        patch_name = self.patch_names[idx]
        image_path = os.path.join(self.tissue_dir, f'{patch_name}.png')
        instance_path = os.path.join(self.instance_dir, f'{patch_name}.npy')

        # Image: HWC, RGB, float in [0, 1].
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f'Failed to read {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        # Instance mask: HWC with C=1.
        instance = np.load(instance_path)
        instance = instance[..., None].astype(np.float32)

        class_mask_np = self._load_class_mask(patch_name)  # [H, W] int64

        # Pack instance + class into a 2-channel HWC mask so SynchronizedTransform
        # applies the same geometric augmentation to both.
        # TODO(cluster-test): verify SynchronizedTransform accepts multi-channel
        # mask input cleanly. If it doesn't, the per-channel split below will
        # surface the issue at first cluster run; the fix is to apply the same
        # geometric ops to the class mask manually using transform parameters.
        combined = np.concatenate(
            [instance, class_mask_np[..., None].astype(np.float32)], axis=-1,
        )

        if self.transform is not None:
            image, combined = self.transform(image=image, mask=combined)
            # SynchronizedTransform may return mask as CHW tensor or HWC numpy
            # depending on its internal pipeline. Normalize to HWC numpy.
            if isinstance(combined, torch.Tensor):
                if combined.dim() == 3 and combined.shape[0] == 2:
                    combined = combined.permute(1, 2, 0).cpu().numpy()
                else:
                    combined = combined.cpu().numpy()

        instance = combined[..., 0:1]                       # [H, W, 1] float
        class_mask_np = combined[..., 1].astype(np.int64)   # [H, W] int

        # Defensive: if the transform returned a numpy image, lift to CHW tensor.
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        h_map, v_map = self._calculate_distance_maps(instance)
        binary_mask = self._process_mask_binary(instance)
        mask_2ch = np.stack([binary_mask, 1 - binary_mask], axis=2).astype(np.float32)
        distance_map = np.stack([v_map, h_map], axis=2).astype(np.float32)

        mask_2ch_t = self.transform_fin(image=mask_2ch)['image']
        distance_map_t = self.transform_fin(image=distance_map)['image']
        instance_mask_t = self.transform_fin(image=instance)['image']
        class_mask_t = torch.from_numpy(class_mask_np).long()

        return image, mask_2ch_t, distance_map_t, instance_mask_t, class_mask_t
