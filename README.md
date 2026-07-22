# Snake AI

Klassisches Snake-Spiel + zwei unabhängige selbstlernende KIs (Neuroevolution & Deep Q-Learning).
Beide teilen sich nur Werkzeuge (Spiel-Engine, Wahrnehmung, Netz-Grundgerüst) — niemals ihr Wissen.

Siehe `CLAUDE.md` für vollständigen Projektkontext.

## Setup

```bash
source venv/bin/activate          # Mac   (Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

## Starten

```bash
python play_human.py              # selbst spielen (Pfeiltasten / WASD)

python train_evolution.py         # KI 1: Neuroevolution, Dashboard
python train_dqn.py               # KI 2: Deep Q-Learning, Live-Dashboard
python train_dqn.py --headless 30 # dasselbe ohne Fenster, 30 Minuten (schnellster Lernmodus)
python train_dqn.py --headless 30 --weiter   # auf dem gespeicherten Champion aufbauen

python watch_ai.py                # dem Neuroevolution-Champion zuschauen
python watch_ai.py dqn            # dem DQN-Champion zuschauen
```

DQN braucht PyTorch und läuft deshalb nur unter dem normalen CPython-venv, **nicht** unter PyPy.
Die Neuroevolution kommt ohne PyTorch aus und profitiert von PyPy (siehe `PYPY_SETUP.md`).

## Wo Daten landen

| Pfad | Inhalt |
|---|---|
| `models/dqn_champion.pt` | bester DQN-Bot (nach Prüfungs-Ø, nicht nach Glückspartie) |
| `models/evo_champion.pt` | bester Neuroevolution-Bot |
| `logs/dqn-<zeit>.csv` | Verlauf: Score, Prüfung, Loss, Effizienz, Todesursachen |
| `logs/dqn-<zeit>-config.json` | alle Einstellungen dieses Laufs (für faire Vergleiche) |

`models/` und `logs/` sind gitignored.

## Stellschrauben

Alles an einem Ort: `ai/dqn/config.py` (Belohnungen, Puffer, Lernrate, Netzgröße, Wahrnehmung …).
Die wichtigsten davon lassen sich auch direkt im Startmenü des DQN-Fensters umstellen.
