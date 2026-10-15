"""All experiment paths are confined to this independent revision directory."""
from pathlib import Path
import hashlib
import json
import platform
import random
import numpy as np
import torch

def canonical(path):
    value=str(Path(path).resolve())
    if value.startswith("\\\\?\\UNC\\"):
        value="\\\\"+value[8:]
    elif value.startswith("\\\\?\\"):
        value=value[4:]
    return Path(value)

ROOT = canonical(__file__).parents[1]

def local_path(value):
    path = canonical(ROOT / value)
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside revision directory: {path}")
    return path

def write_json(path, value):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def environment():
    import pandas, scipy, sklearn
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, pandas=pandas.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__, device="cpu")

def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
