
# ml/learning/lib/data.py

import math

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Sampler, TensorDataset


def class_mapping(labels):
    classes = np.unique(np.asarray(labels))
    if not len(classes):
        raise ValueError("Cannot create a class mapping from empty labels.")

    return classes, {
        str(label): i for i, label in enumerate(classes)
    }


def split_indices(labels, validation_fraction=0.1, seed=42, domains=None):
    labels = np.asarray(labels)
    indices = np.arange(len(labels))

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if validation_fraction == 0:
        return indices, None

    if domains is not None:
        domains = np.asarray(domains)
        if len(domains) != len(labels):
            raise ValueError("domains and labels must have the same length.")

        strata = [
            f"{d}|{y}" for d, y in zip(domains, labels)
        ]
    else:
        strata = labels

    try:
        return train_test_split(
            indices,
            test_size=validation_fraction,
            stratify=strata,
            random_state=seed,
        )
    except ValueError:
        if domains is None:
            raise

        return train_test_split(
            indices,
            test_size=validation_fraction,
            stratify=labels,
            random_state=seed,
        )


class DomainBalancedBatchSampler(Sampler):
    """Sample equally sized groups from distinct domains."""

    def __init__(self, domains, batch_size, domains_per_batch=4, seed=42):
        domains = np.asarray(domains)

        if domains.ndim != 1 or not len(domains):
            raise ValueError("domains must be a nonempty 1D array.")
        if batch_size < 1 or domains_per_batch < 1:
            raise ValueError(
                "batch_size and domains_per_batch must be positive."
            )

        _, domain_ids = np.unique(domains, return_inverse=True)

        self.groups = [
            np.flatnonzero(domain_ids == d)
            for d in np.unique(domain_ids)
        ]

        self.n_domains = min(
            int(domains_per_batch), len(self.groups)
        )
        self.per_domain = int(batch_size) // self.n_domains

        if self.n_domains < 2 or self.per_domain < 2:
            raise ValueError(
                "Require at least 2 domains and "
                "2 samples per domain."
            )

        self.batch_size = self.n_domains * self.per_domain
        self.n_batches = math.ceil(
            len(domains) / self.batch_size
        )
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            selected = self.rng.choice(
                len(self.groups),
                self.n_domains,
                replace=False,
            )

            batch = np.concatenate([
                self.rng.choice(
                    self.groups[d],
                    self.per_domain,
                    replace=len(self.groups[d]) < self.per_domain,
                )
                for d in selected
            ])

            yield batch.tolist()

    def __len__(self):
        return self.n_batches


def make_loader(group, class_to_idx, batch_size=64, partition="train",
                transform=None, shuffle=False, domain_balanced=False,
                domains_per_batch=4, num_workers=0, seed=42):

    partitions = np.asarray(group.partitions)
    mask = (
        np.ones(len(group.y), dtype=bool)
        if partition is None else partitions == partition
    )

    if not mask.any():
        raise ValueError(
            f"No samples for partition '{partition}'."
        )

    X = np.asarray(group.X)[mask]
    labels = np.asarray(group.y)[mask]
    domains = np.asarray(group.elementary_domains)[mask]

    y = np.asarray([
        class_to_idx[str(v)] for v in labels
    ], dtype=np.int64)

    # ----------------------------------------------------
    # Dataset
    # ----------------------------------------------------

    if transform is not None:
        from vision.lib.dataset import ImageDataset

        dataset = ImageDataset(
            X, labels, domains, class_to_idx, transform
        )
    else:
        if X.dtype.kind in "USO":
            raise ValueError(
                "Non-numeric inputs require an image transform."
            )

        _, domain_ids = np.unique(
            domains, return_inverse=True
        )

        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.long),
            torch.as_tensor(domain_ids, dtype=torch.long),
        )

    generator = torch.Generator().manual_seed(seed)

    # ----------------------------------------------------
    # Domain-balanced sampling
    # ----------------------------------------------------

    if domain_balanced:
        sampler = DomainBalancedBatchSampler(
            domains,
            batch_size,
            domains_per_batch,
            seed,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            generator=generator,
        )

    # ----------------------------------------------------
    # Standard sampling
    # ----------------------------------------------------

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
