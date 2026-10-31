
# vision/lib/dataset.py

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, paths, labels, domains, class_to_idx, transform):
        self.paths = paths
        self.labels = [class_to_idx[str(y)] for y in labels]
        self.domains = domains
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            image = self.transform(image.convert("RGB"))

        return image, self.labels[idx], str(self.domains[idx])


def make_image_loader(group, class_to_idx, transform, batch_size=32,
                      shuffle=False, num_workers=0, partition=None):
    mask = (
        np.ones(len(group.y), dtype=bool) if partition is None
        else group.partitions == partition
    )

    dataset = ImageDataset(
        group.X[mask],
        group.y[mask],
        group.elementary_domains[mask],
        class_to_idx,
        transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
