"""Util kecil: tulis JSON hasil (konversi tipe numpy) — satu file per langkah."""

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"tak bisa serialisasi {type(o)}")


def save_json(name: str, obj: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=_to_jsonable), encoding="utf-8")
    print(f"[hasil] {path}")
    return path
