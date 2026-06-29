"""Load / resolve / save the per-channel YAML config.

All path keys in the config are relative to the repo root, which is taken to be
the parent directory of the config file's folder (e.g. ``<root>/vis/config.yaml``
=> repo root ``<root>``).
"""
import os
import yaml


def load_config(path):
    """Load a channel config and stamp it with resolved absolute paths."""
    path = os.path.abspath(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = path
    cfg["_repo_root"] = os.path.dirname(os.path.dirname(path))
    return cfg


def resolve(cfg, key):
    """Return an absolute path for a path-valued config key."""
    return os.path.join(cfg["_repo_root"], cfg[key])


def save_config(cfg, path=None):
    """Persist the config back to disk, dropping internal ``_`` keys."""
    path = path or cfg["_path"]
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
