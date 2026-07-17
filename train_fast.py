#!/usr/bin/env python3
"""Launcher für schnelles Training -- automatisch headless unter PyPy3.

Hintergrund: Weder pygame noch PyTorch haben fertige PyPy3-Wheels unter
Windows (beide muessten aus C/C++-Quellcode gebaut werden, wofuer man
Visual-Studio-Build-Tools braeuchte -- unnoetiger Aufwand). Das Training
selbst (ai/evolution/train_evolution.py) braucht aber gar kein pygame/Torch --
nur reines NumPy. Deshalb laeuft train_fast.py unter PyPy3 automatisch
HEADLESS: kein Fenster, nur Konsolen-Ausgabe + CSV-Log, und der Champion wird
als Genom (.npy) + Metadaten (.json) gespeichert (siehe build_champion.py, um
daraus spaeter die .pt-Datei fuers Zuschauen zu bauen).

Unter normalem CPython (das venv, mit pygame+Torch) startet stattdessen wie
gewohnt das Dashboard mit Live-Fenster.

Verwendung:
    pypy3 train_fast.py                       (headless, maximale Geschwindigkeit)
    pypy3 train_fast.py --generations 500      (nach 500 Generationen stoppen)
    python train_fast.py                       (Dashboard-Fenster, normal schnell)

Danach zuschauen (immer mit normalem Python, braucht pygame+Torch):
    python build_champion.py    (nur noetig, falls mit PyPy trainiert wurde)
    python watch_ai.py
"""

from __future__ import annotations

import argparse
import platform


def _is_pypy() -> bool:
    return platform.python_implementation() == "PyPy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations", type=int, default=100_000,
        help="Anzahl Generationen (Standard: praktisch endlos, Strg+C zum Stoppen)",
    )
    args = parser.parse_args()

    if _is_pypy():
        print("[OK] PyPy3 erkannt -- Training laeuft HEADLESS (kein Fenster, kein pygame/Torch noetig)")
        print("     Champion wird laufend als Genom+Metadaten gespeichert (CSV-Log wie gewohnt).")
        print("     Danach zum Zuschauen: 'python build_champion.py' dann 'python watch_ai.py'.\n")
        from ai.evolution.train_evolution import main as headless_main
        try:
            headless_main(generations=args.generations)
        except KeyboardInterrupt:
            print("\nTraining gestoppt (Strg+C). Champion bleibt gespeichert.")
    else:
        print("Nutze CPython mit Dashboard-Fenster.")
        print("Fuer 5-10x Speedup: PyPy3 installieren (siehe PYPY_SETUP.md), dann 'pypy3 train_fast.py'.\n")
        from dashboard.live_view import main as dashboard_main
        dashboard_main()


if __name__ == "__main__":
    main()
