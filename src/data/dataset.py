from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):
    def __init__(self, data_dir, split, lr_size):
        self.hr_dir = Path(data_dir) / split / "HR"
        self.lr_dir = Path(data_dir) / split / f"LR_{lr_size}"
        self.images = sorted(self.hr_dir.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        hr_path = self.images[index]
        lr_path = self.lr_dir / hr_path.name

        hr = cv2.imread(str(hr_path))
        lr = cv2.imread(str(lr_path))

        lr = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        hr = torch.from_numpy(hr.transpose(2, 0, 1).copy()).float() / 255.0
        lr = torch.from_numpy(lr.transpose(2, 0, 1).copy()).float() / 255.0

        return {"input": lr, "target": hr}


class DenoisingDataset(Dataset):
    def __init__(self, data_dir, split, sigma):
        self.images = sorted((Path(data_dir) / split / "HR").glob("*.png"))
        self.split = split
        self.sigma = sigma

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.split == "val":
            rng = np.random.default_rng(42 + index)
        else:
            rng = np.random

        noise = rng.normal(0, self.sigma, image.shape).astype(np.float32)
        noisy = np.clip(image + noise, 0, 1)

        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1).copy())
        return {"input": noisy, "target": image}
