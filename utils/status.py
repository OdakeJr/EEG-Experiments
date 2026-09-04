# utils/status.py

import hashlib
import json

from utils.storage import exists, load_manifest


def make_signature(params):
    text = json.dumps(params, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()


def is_done(manifest_path, params):
    if not exists(manifest_path):
        return False

    manifest = load_manifest(manifest_path)

    return (
        manifest.get("status") == "done"
        and manifest.get("signature") == make_signature(params)
    )


def make_manifest(status, params, execution_time=None, error=None):
    return {
        "status": status,
        "params": params,
        "signature": make_signature(params),
        "execution_time": execution_time,
        "error": error,
    }