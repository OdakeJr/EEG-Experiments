import argparse
import hashlib
import json
from pathlib import Path


OUTPUT_ROOT = Path("outputs/training")


def make_signature(params):
    text = json.dumps(params, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()


def replace_paths(value, old_path, new_path):
    if isinstance(value, dict):
        return {k: replace_paths(v, old_path, new_path) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_paths(v, old_path, new_path) for v in value]
    if isinstance(value, str) and value.startswith(old_path):
        return new_path + value[len(old_path):]
    return value


def migrate_manifest(manifest_path, apply=False):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "done":
        return "skip"

    params = manifest.get("params", {})
    training_params = params.get("training_params", {})

    if "device" not in training_params:
        return "skip"

    old_dir = manifest_path.parent
    old_signature = manifest["signature"]
    device = training_params.pop("device")
    new_signature = make_signature(params)

    old_suffix = f"_{old_signature[:12]}"
    if not old_dir.name.endswith(old_suffix):
        raise RuntimeError(f"Unexpected folder name: {old_dir}")

    name = old_dir.name[:-len(old_suffix)]
    new_dir = old_dir.with_name(f"{name}_{new_signature[:12]}")

    if new_dir.exists() and new_dir != old_dir:
        raise RuntimeError(f"Destination already exists: {new_dir}")

    manifest["signature"] = new_signature
    manifest.setdefault("runtime", {})["device"] = device
    manifest = replace_paths(manifest, str(old_dir), str(new_dir))

    print(f"{old_dir}")
    print(f"  device: {device}")
    print(f"  {old_signature[:12]} -> {new_signature[:12]}")
    print(f"  -> {new_dir}")

    if apply:
        manifest_path.write_text(json.dumps(manifest, indent=2))
        old_dir.rename(new_dir)

    return "migrated"


def main(apply=False):
    manifests = list(OUTPUT_ROOT.rglob("manifest.json"))
    migrated = skipped = 0

    for manifest_path in manifests:
        result = migrate_manifest(manifest_path, apply)
        if result == "migrated":
            migrated += 1
        else:
            skipped += 1

    mode = "APPLIED" if apply else "DRY RUN"
    print(f"\n{mode} | migrated={migrated} | skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    main(args.apply)