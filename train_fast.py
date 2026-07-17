#!/usr/bin/env python3
"""Launcher für PyPy3-beschleunigtes Training.

PyPy3 ist ein Drop-in-Replacement für CPython mit JIT-Compilation.
Erwarteter Speedup: 5-10x schneller als CPython (besonders bei langen Generationen).

Verwendung:
    python train_fast.py        (versucht PyPy3, fällt auf CPython zurück)
    pypy3 train_fast.py         (explizit PyPy3)

Erst müsste PyPy3 installiert sein:
    Mac/Linux:   pip install pypy3
    Windows:     siehe https://www.pypy.org/download.html (oder choco install pypy3)

Danach:
    pypy3 -m pip install numpy torch   (PyPy-Umgebung aktualisieren)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def main():
    # Versuche PyPy3 zu nutzen, falls vorhanden
    try:
        # Prüfe ob pypy3 verfügbar ist
        result = subprocess.run(
            ["pypy3", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print(f"✓ Nutze PyPy3 (5-10x schneller als CPython)")
            print(f"  {result.stdout.strip()}\n")
            # Starte Training mit PyPy3
            subprocess.run([sys.executable, str(PROJECT_ROOT / "train_evolution.py")])
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print("ℹ PyPy3 nicht gefunden — nutze CPython (langsamer)")
    print("  Für Speed-Up: pip install pypy3")
    print("             dann: pypy3 -m pip install -r requirements.txt")
    print("             dann: pypy3 train_fast.py\n")
    # Fallback auf normales Python
    subprocess.run([sys.executable, str(PROJECT_ROOT / "train_evolution.py")])


if __name__ == "__main__":
    main()
