
# vision/dataset_handlers/pacs.py

import shutil
from pathlib import Path

import pandas as pd
from datasets import Image, load_dataset


DATASET_NAME = "pacs"
HF_DATASET = "flwrlabs/pacs"


def prepare_pacs(params):
    root = params.get("root")
    if root is None:
        raise ValueError("'root' must be specified for PACS.")

    root = Path(root).expanduser().resolve()
    dataset = load_dataset(HF_DATASET, split="train")
    dataset = dataset.cast_column("image", Image(decode=False))
    classes = dataset.features["label"].names

    records = []

    for i, sample in enumerate(dataset):
        image = sample["image"]
        suffix = Path(image["path"] or "").suffix or ".jpg"
        path = root / sample["domain"] / f"{i:05d}{suffix}"

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if image["bytes"] is not None:
                path.write_bytes(image["bytes"])
            else:
                shutil.copyfile(image["path"], path)

        records.append({
            "dataset": DATASET_NAME,
            "domain": sample["domain"],
            "sample_index": i,
            "label": classes[int(sample["label"])],
            "image_path": str(path),
        })

    data = pd.DataFrame(records)

    info = {
        "dataset_name": DATASET_NAME,
        "representation": "image",
        "input_shape": None,
        "feature_columns": None,
        "label_column": "label",
        "domain_columns": ["dataset", "domain"],
        "metadata_columns": ["sample_index", "image_path"],
        "metadata": {
            "n_samples": len(data),
            "n_domains": data["domain"].nunique(),
            "classes": classes,
        },
    }

    return data, info
