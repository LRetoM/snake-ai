# Snake AI Projekt — Kontext für Claude Code

Dies ist ein Lernprojekt. Der User (Luca) kommt aus TypeScript/SharePoint-Entwicklung,
ist aber neu bei Python und komplett neu bei KI/Machine Learning/Reinforcement Learning.
**Erklär Konzepte einfach und mit Alltagsvergleichen, wenn neue KI-Begriffe auftauchen.**
Kommuniziere auf Deutsch, da der User Deutsch schreibt.

## Ziel des Projekts

1. Ein klassisches Snake-Spiel bauen (komplett, alle üblichen Features), spielbar mit Pfeiltasten.
2. Danach ZWEI unabhängige, separate KIs bauen, die das Spiel selbst lernen:
   - **Neuroevolution** (zuerst): 100 Schlangen parallel, genetischer Algorithmus,
     Generationen die sich verbessern, live im Dashboard beobachtbar.
   - **Deep Q-Learning / DQN** (danach): eine Schlange, klassisches Reinforcement Learning
     mit Belohnung/Bestrafung, Experience Replay.
3. Die zwei KIs **arbeiten NICHT zusammen** und teilen kein Wissen. Es sind zwei komplett
   getrennte, unabhängige Bots, die am Ende verglichen werden können (wer spielt besser,
   wer lernt schneller). Sie teilen sich nur die Spiel-Engine, das Netz-Grundgerüst
   und das Dashboard (Werkzeuge), nicht ihr gelerntes Wissen.
4. **Live-Dashboard ist Pflicht**: der User will während des Trainings live sehen was
   passiert — aktuelle Generation, Scores, Lernkurve, die besten laufenden Runs.

## Sehr wichtige Leitplanke: KI lernt "from scratch"

**Strikte Trennung zwischen Spiel (Umgebung) und KI (Agent):**

- Das Spiel legt ALLE Regeln fest und die KI hat darauf keinen Zugriff/Einfluss:
  Wo die Frucht spawnt, Geschwindigkeit, Kollisionsregeln, Bewegungslogik.
  Das ist reiner, fest programmierter Spielcode — ändert sich nie.
- Die KI bekommt AUSSCHLIESSLICH:
  - **Input**: eine Wahrnehmung des aktuellen Zustands (z.B. Gefahr links/rechts/geradeaus,
    Richtung der Frucht relativ zur Schlange, aktuelle Bewegungsrichtung) — vergleichbar
    damit, dass ein Mensch nur den Bildschirm sieht, nicht den Code.
  - **Output**: eine Aktion (geradeaus / links abbiegen / rechts abbiegen)
  - **Feedback**: Score / Tod / Überlebenszeit als Lernsignal
- Die KI kennt am Anfang KEINE Regeln (nicht mal "Wand = tödlich" oder "Frucht = gut").
  Sie startet mit zufälligem Gehirn, crasht anfangs ständig, und lernt rein durch
  Wiederholung/Versuch-und-Irrtum.
- **Niemals Spielstrategie in den KI-Code hardcoden.** Kein Pathfinding, keine
  Heuristiken die der KI die Lösung vorgeben. Die KI muss alles selbst herausfinden.

## Sprache & Stack

Alles in Python, ein durchgehender Stack (kein Sprachmix):
- **Spiel**: pygame
- **Neuronale Netze (beide KIs)**: PyTorch
- **Dashboard**: pygame (mitgezeichnet) oder separates Fenster, noch zu entscheiden

## Geplante Architektur

```
snake-ai/
├── game/
│   ├── snake_game.py      Kern-Spiellogik (Bewegung, Kollision, Frucht-Spawn, step())
│   └── renderer.py        Zeichnen mit pygame
├── play_human.py          Mensch spielt selbst mit Pfeiltasten
│
├── ai/
│   ├── perception.py      Spielzustand -> Zahlen-Vektor (Wahrnehmung, von BEIDEN KIs genutzt)
│   ├── network.py         Neuronales Netz Grundgerüst (von BEIDEN KIs genutzt)
│   │
│   ├── evolution/         Neuroevolution (Phase 3a)
│   │   ├── population.py    Population von z.B. 100 Netzen
│   │   ├── genetics.py      Selektion, Crossover, Mutation
│   │   └── train_evolution.py
│   │
│   └── dqn/               Deep Q-Learning (Phase 3b)
│       ├── agent.py         Q-Learning Logik, Exploration/Exploitation
│       ├── memory.py        Experience Replay Buffer
│       └── train_dqn.py
│
├── dashboard/
│   └── live_view.py        Generation, Scores, Lernkurve, mehrere Snakes live anzeigen
│
├── models/                 Gespeicherte trainierte Netze (.pt, gitignored)
└── logs/                   Trainingsverlauf/CSV-Logs (gitignored)
```

## Bau-Reihenfolge (jeder Schritt baut auf dem vorherigen auf)

1. `snake_game.py` + `renderer.py` + `play_human.py` — klassisches Snake, komplett spielbar
2. `step(action)`-Schnittstelle + Headless-Modus (ohne Fenster, für schnelles Training) + `perception.py`
3. `network.py` — leeres Netz-Grundgerüst (Input ~11 Zahlen -> Hidden Layer -> 3 Outputs)
4. `ai/evolution/` + Dashboard — Neuroevolution läuft, 100 Snakes, Generationen, live sichtbar
5. `ai/dqn/` — DQN läuft, nutzt dieselbe Umgebung/Netz-Basis/Dashboard, eigene Lernschleife

**Aktueller Stand: Projektordner + venv + Pakete sind eingerichtet. Schritt 1 (klassisches
Snake-Spiel) ist der nächste zu bauende Teil — noch nicht begonnen.**

## Setup

```bash
cd ~/Documents/snake-ai
source venv/bin/activate   # venv ist bereits erstellt
python play_human.py       # sobald Schritt 1 existiert
```

Installierte Pakete (venv bereits vorhanden, nicht neu erstellen): pygame, numpy, torch, matplotlib
