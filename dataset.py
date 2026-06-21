from importlib.resources import path

import cv2
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms
from typing import Union


class StripeAugmentor:
    """Procedurally adds random black transparent stripes to an image."""
    def __init__(self, min_alpha=0.1, max_alpha=0.8, min_stripes=1, max_stripes=8):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_stripes = min_stripes
        self.max_stripes = max_stripes

    def add_stripes(
        self,
        input_img: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        input_img:
            - numpy array (H, W, 3)
            - torch tensor (H, W, 3) or (3, H, W)

        Returns:
            Same type as input.
        """
        input_is_tensor = isinstance(input_img, torch.Tensor)
        if input_is_tensor:
            original_device = input_img.device
            original_dtype = input_img.dtype

            img = input_img.detach().cpu()

            # CHW -> HWC
            chw_input = False
            if img.ndim == 3 and img.shape[0] == 3:
                chw_input = True
                img = img.permute(1, 2, 0)

            img = img.numpy()
        else:
            img = input_img

        h, w = img.shape[:2]
        stained = img.copy().astype(np.float32)
        num_stripes = random.randint(self.min_stripes, self.max_stripes)
        combined_mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_stripes):
            width = random.randint(5, 40)
            alpha = random.uniform(self.min_alpha, self.max_alpha)
            angle = random.uniform(0, 360)

            diag = int(np.sqrt(h**2 + w**2))
            m_size = diag * 3

            best_mask = None

            for _ in range(20):
                cx = random.randint(0, w)
                cy = random.randint(0, h)
                m = np.zeros((m_size, m_size), dtype=np.float32)

                x1 = m_size // 2 - width // 2
                cv2.rectangle(
                    m,
                    (x1, 0),
                    (x1 + width, m_size),
                    1.0,
                    -1
                )
                matrix = cv2.getRotationMatrix2D(
                    (m_size // 2, m_size // 2),
                    angle,
                    1.0
                )
                m = cv2.warpAffine(m, matrix, (m_size, m_size))

                x_s = m_size // 2 - cx
                y_s = m_size // 2 - cy

                candidate_mask = m[y_s:y_s+h, x_s:x_s+w]
                overlap = np.sum(combined_mask * candidate_mask)

                if overlap < 1.0:
                    best_mask = candidate_mask
                    break

            if best_mask is not None:
                combined_mask += best_mask
                stained *= (1.0 - alpha * best_mask[:, :, None])

        # ==========================================
        # Return tensor in original dtype/range
        # ==========================================
        if input_is_tensor:
            result = torch.from_numpy(stained)
            if chw_input:
                result = result.permute(2, 0, 1)
            result = result.to(
                device=original_device,
                dtype=original_dtype
            )
            return result

        # ==========================================
        # Return numpy image as uint8
        # ==========================================
        stained = np.clip(stained, 0, 255).astype(np.uint8)
        return stained

class MangaStripeDataset(Dataset):
    def __init__(self, img_dir, patch_size=256, transform=None, training=True):
        self.img_paths = self._get_img_paths(img_dir)
        self.patch_size = patch_size
        self.augmentor = StripeAugmentor()
        self.training = training
        self.transform = transform

    def _read_img(self, path):
        return Image.open(path).convert("RGB")
    
    def _get_img_paths(self, dir):
        exts = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        paths = []
        for ext in exts:
            paths.extend(glob(f"{dir}/*.{ext}"))
        return paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self._read_img(self.img_paths[idx])

        if self.transform is not None:
            # Clean image tensor (C,H,W), float32, [0,1]
            target_patch: torch.Tensor = self.transform(img)

            # Create corrupted version
            stained_patch: torch.Tensor = self.augmentor.add_stripes(target_patch.clone())
        else:
            stained_np = self.augmentor.add_stripes(img)
            target_patch = (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            stained_patch = (torch.from_numpy(stained_np).permute(2, 0, 1).float()/ 255.0)

        return stained_patch, target_patch