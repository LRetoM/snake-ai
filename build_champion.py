"""Baut models/evo_champion.pt aus dem gespeicherten Genom -- nur noetig, wenn
mit PyPy3 trainiert wurde (dort gibt es kein PyTorch, siehe PYPY_SETUP.md).

Beim Training mit normalem CPython (venv) passiert das automatisch -- dieses
Skript brauchst du nur, wenn train_fast.py unter PyPy3 lief und du danach
watch_ai.py / das Dashboard mit dem trainierten Champion nutzen willst.

Verwendung (mit normalem Python, das PyTorch installiert hat):
    source venv/bin/activate      (Mac)   /   venv\\Scripts\\activate   (Windows)
    python build_champion.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")
_GENOME_PATH = os.path.join(_MODEL_DIR, "evo_champion_genome.npy")
_META_PATH = os.path.join(_MODEL_DIR, "evo_champion_meta.json")


def main() -> None:
    if not os.path.exists(_GENOME_PATH) or not os.path.exists(_META_PATH):
        print(
            "Kein gespeicherter Champion gefunden.\n"
            f"  Erwartet: {_GENOME_PATH}\n"
            f"        und {_META_PATH}\n\n"
            "Erst trainieren, z.B.: pypy3 train_fast.py"
        )
        sys.exit(1)

    from ai.torch_bridge import save_champion_checkpoint

    genome = np.load(_GENOME_PATH)
    with open(_META_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    path = save_champion_checkpoint(genome, meta, _MODEL_DIR)
    print(f"[OK] {path} gebaut (Ø-Score {meta['score']:.2f}, Generation {meta['generation']}).")
    print("Jetzt kannst du z.B. 'python watch_ai.py' starten.")


if __name__ == "__main__":
    main()
