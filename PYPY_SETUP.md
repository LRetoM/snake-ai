# PyPy3 — 5-10x schnelleres Training

PyPy3 ist ein Drop-in-Replacement für CPython mit JIT-Compilation (Just-In-Time). 
Für dieses Projekt erwarten wir **5-10x Speedup** beim Training.

## Installation

### Mac / Linux
```bash
pip install pypy3
```

### Windows
Entweder:
```bash
choco install pypy3
```
Oder manuell von https://www.pypy.org/download.html herunterladen.

## Setup der PyPy-Umgebung

Nach Installation, einmal die Dependencies in PyPy installieren:

```bash
pypy3 -m pip install -r requirements.txt
```

Das ist ein One-Time Setup. PyPy hat seine eigene isolierte Umgebung.

## Training starten

### Einfach (Auto-Detection):
```bash
python train_fast.py
```
Das versucht PyPy3 zu nutzen, fällt automatisch auf CPython zurück falls nicht installiert.

### Explizit PyPy3:
```bash
pypy3 train_fast.py
```

## Speedup-Beispiel

**CPython (Standard):**
```
Gen 127 (Score 206):  6.89 Gen/s
```

**PyPy3 (mit JIT):**
```
Gen 127 (Score 206):  40-60+ Gen/s erwartet  
```

Die Beschleunigung ist besonders bei **langen Generationen** sichtbar (hohe Scores, lange Schlangen).

## Wichtig

- PyPy ist **vollständig kompatibel** mit diesem Code — keine Änderungen nötig
- Die trainierte KI sieht genau gleich aus (same NumPy, same PyTorch)
- `models/evo_champion.pt` funktioniert mit CPython und PyPy gleich
- Nur das Training ist schneller; alles andere (watch_ai.py, play_human.py) läuft auch mit PyPy

## Fallback

Wenn PyPy nicht installiert ist oder Probleme bereitet: Einfach das Standard-Training nutzen
```bash
python train_evolution.py
```
Das läuft dann mit CPython (normal, kein Speedup, aber alles funktioniert).
