"""Wahrnehmung: verwandelt den Spielzustand in einen Zahlen-Vektor fuer die KI.

Das ist die "Sinnesorgan"-Schicht. Die KI bekommt NIE das ganze Spielfeld oder
gar den Code -- sie bekommt nur diese wenigen Zahlen, so wie ein Mensch nur den
Bildschirm sieht (und nicht den Programmcode dahinter).

Von BEIDEN KIs benutzt (Neuroevolution und DQN), aber beide lernen daraus voellig
eigenstaendig -- geteilt wird nur das Werkzeug, nicht das Wissen.

Ganz wichtige Leitplanke:
Diese Datei enthaelt KEINE Strategie. Sie sagt der KI nicht, was sie tun soll --
sie beschreibt nur, was gerade "zu sehen" ist:
  - Gefahr direkt geradeaus / rechts / links
  - in welche Richtung die Schlange gerade schaut
  - in welcher groben Richtung die naechste Frucht liegt
Was die KI daraus macht, muss sie selbst herausfinden. Kein Pathfinding, keine
Heuristik, die den Weg vorgibt.

Der Vektor hat 11 Zahlen (jeweils 0.0 oder 1.0). Genau diese 11 werden spaeter
zum Eingang des neuronalen Netzes (network.py, Schritt 3).
"""

from __future__ import annotations

import numpy as np

from game.snake_game import Action, Direction, SnakeGame, relative_turn

# Anzahl der Eingangswerte -- das neuronale Netz wird genau so viele Eingaenge haben.
INPUT_SIZE = 11

# Klartext-Namen der 11 Werte -- nur zum Anschauen/Debuggen, nicht fuer die KI.
FEATURE_LABELS = [
    "Gefahr geradeaus",
    "Gefahr rechts",
    "Gefahr links",
    "Richtung: oben",
    "Richtung: rechts",
    "Richtung: unten",
    "Richtung: links",
    "Frucht ist links",
    "Frucht ist rechts",
    "Frucht ist oben",
    "Frucht ist unten",
]


def _nearest_fruit(game: SnakeGame):
    """Die der Schlange am naechsten liegende Frucht (Manhattan-Distanz).

    Bei mehreren Fruechten "schaut" die KI auf die naechste -- das ist ein reiner
    Sinneseindruck (welcher Apfel ist am naechsten), keine Wegplanung.
    """
    hx, hy = game.head
    return min(game.fruits, key=lambda f: abs(f[0] - hx) + abs(f[1] - hy))


def perceive(game: SnakeGame) -> np.ndarray:
    """Baut den 11er-Wahrnehmungsvektor aus dem aktuellen Spielzustand.

    Rueckgabe: numpy-Array mit 11 float32-Werten (0.0/1.0).
    """
    head_x, head_y = game.head
    direction = game.direction

    # --- 1) Gefahr geradeaus / rechts / links --------------------------------
    # Fuer jede der drei relativen Aktionen: Wohin kaeme der Kopf? Ist das toedlich?
    # is_deadly() lebt im Spiel (dort liegen die Regeln) -- wir fragen nur ab.
    danger = []
    for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
        move_dir = relative_turn(direction, action)
        dx, dy = move_dir.value
        next_cell = (head_x + dx, head_y + dy)
        danger.append(1.0 if game.is_deadly(next_cell) else 0.0)

    # --- 2) Aktuelle Blickrichtung (one-hot) ---------------------------------
    dir_up = 1.0 if direction == Direction.UP else 0.0
    dir_right = 1.0 if direction == Direction.RIGHT else 0.0
    dir_down = 1.0 if direction == Direction.DOWN else 0.0
    dir_left = 1.0 if direction == Direction.LEFT else 0.0

    # --- 3) Grobe Richtung zur naechsten Frucht ------------------------------
    fruit_x, fruit_y = _nearest_fruit(game)
    food_left = 1.0 if fruit_x < head_x else 0.0
    food_right = 1.0 if fruit_x > head_x else 0.0
    food_up = 1.0 if fruit_y < head_y else 0.0
    food_down = 1.0 if fruit_y > head_y else 0.0

    return np.array(
        [
            danger[0], danger[1], danger[2],
            dir_up, dir_right, dir_down, dir_left,
            food_left, food_right, food_up, food_down,
        ],
        dtype=np.float32,
    )


def describe(vector: np.ndarray) -> str:
    """Menschenlesbare Beschreibung eines Wahrnehmungsvektors (fuers Verstehen)."""
    lines = [
        f"  {label:<20}: {int(value)}"
        for label, value in zip(FEATURE_LABELS, vector)
    ]
    return "\n".join(lines)
