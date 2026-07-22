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
    # danger_flags() lebt im Spiel (dort liegen die Regeln) -- wir fragen nur ab.
    # Alle 3 Kandidatenzellen auf einmal abfragen (statt 3x is_deadly() einzeln)
    # spart bei langen Schlangen spuerbar Zeit, siehe Kommentar dort.
    candidate_cells = []
    for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
        move_dir = relative_turn(direction, action)
        dx, dy = move_dir.value
        candidate_cells.append((head_x + dx, head_y + dy))
    danger = [1.0 if flag else 0.0 for flag in game.danger_flags(candidate_cells)]

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


# =========================================================================== #
# REICHERE WAHRNEHMUNG ("rich") -- nur fuer das DQN, optional
# =========================================================================== #
# Warum ueberhaupt eine zweite Wahrnehmung?
#
# Die 11 Zahlen oben sind ehrlich, aber SEHR grob. "Gefahr geradeaus" ist nur
# dann 1, wenn die Gefahr GENAU eine Zelle vor dem Kopf liegt. Ob zwei Zellen
# weiter eine Wand kommt oder das Feld noch 15 Zellen offen ist -- identische
# Wahrnehmung. Fuer die KI sehen tausende voellig verschiedene Spielsituationen
# exakt gleich aus (Fachbegriff: "State Aliasing"). Sie kann dann gar nicht
# besser werden, egal wie lange sie trainiert -- so wie ein Mensch, der Snake
# durch einen Strohhalm spielt.
#
# Die reiche Wahrnehmung gibt ihr mehr von dem, was ein Mensch auf dem
# Bildschirm ohnehin SIEHT: wie weit ist die Wand in jede Richtung, wie weit
# der eigene Koerper, wo liegt die Frucht -- in 8 Richtungen ("Tastsinn" wie
# Schnurrhaare). Das ist immer noch reine Wahrnehmung:
#
#   KEIN Pathfinding, KEINE Bewertung, KEIN Hinweis was zu tun ist.
#   Insbesondere KEIN "wie viel Platz habe ich noch" (Flood-Fill) -- das waere
#   eine ausgerechnete Heuristik, die der KI die Loesung des Einsperr-Problems
#   halb verraet. Sie muss selbst lernen, dass enge Bereiche gefaehrlich sind.
#
# Alles ist EGOZENTRISCH (aus Sicht der Schlange: vorne/rechts/hinten/links)
# statt in Himmelsrichtungen. Das ist der gleiche Trick wie bei den Aktionen:
# eine Situation, die die Schlange nach oben fahrend gelernt hat, gilt dann
# automatisch auch nach unten fahrend -- sie muss alles nur EINMAL lernen
# statt viermal.

# Die 8 Tastrichtungen als (vorne, rechts)-Anteile, im Uhrzeigersinn ab "vorne".
_RAY_OFFSETS = [
    (1, 0),    # vorne
    (1, 1),    # vorne-rechts
    (0, 1),    # rechts
    (-1, 1),   # hinten-rechts
    (-1, 0),   # hinten
    (-1, -1),  # hinten-links
    (0, -1),   # links
    (1, -1),   # vorne-links
]
_RAY_NAMES = ["vorne", "vorne-rechts", "rechts", "hinten-rechts",
              "hinten", "hinten-links", "links", "vorne-links"]

RICH_INPUT_SIZE = 39

RICH_FEATURE_LABELS = (
    [f"Wand {n}" for n in _RAY_NAMES]
    + [f"Körper {n}" for n in _RAY_NAMES]
    + [f"Frucht {n}" for n in _RAY_NAMES]
    + ["Gefahr geradeaus", "Gefahr rechts", "Gefahr links"]
    + ["Frucht vorne/hinten", "Frucht rechts/links"]
    + ["Richtung: oben", "Richtung: rechts", "Richtung: unten", "Richtung: links"]
    + ["Schwanz vorne", "Schwanz rechts", "Schwanz hinten", "Schwanz links"]
    + ["Länge", "Hunger"]
)


def perceive_rich(game: SnakeGame) -> np.ndarray:
    """Reichere Wahrnehmung: 39 Zahlen, egozentrisch. Immer noch reine Sinne.

    Aufbau:
      0-7   Wand-Naehe in 8 Richtungen        (1/Abstand, 1.0 = direkt daneben)
      8-15  Koerper-Naehe in 8 Richtungen     (1/Abstand, 0.0 = nichts in Sicht)
      16-23 Frucht in 8 Richtungen            (1/Abstand, 0.0 = nicht in Sicht)
      24-26 Gefahr geradeaus / rechts / links (0/1, wie in der einfachen Version)
      27-28 Frucht-Versatz vorne-hinten / rechts-links (-1..1, egozentrisch)
      29-32 aktuelle Blickrichtung (one-hot, absolut)
      33-36 wohin der Schwanz zeigt, relativ zum Kopf (one-hot)
      37    Laenge der Schlange (0..1, anteilig am Spielfeld)
      38    Hunger: Schritte seit der letzten Frucht (0..1, bei ~500 gedeckelt)
    """
    head_x, head_y = game.head
    direction = game.direction

    # Vorne = aktuelle Richtung, Rechts = 90 Grad im Uhrzeigersinn davon.
    fx, fy = direction.value
    rx, ry = -fy, fx

    occupied = game.occupied
    fruits = game.fruits
    cols, rows = game.cols, game.rows
    max_len = max(cols, rows)

    wall_feat = [0.0] * 8
    body_feat = [0.0] * 8
    fruit_feat = [0.0] * 8

    for k, (fwd, right) in enumerate(_RAY_OFFSETS):
        dx = fwd * fx + right * rx
        dy = fwd * fy + right * ry
        x, y = head_x, head_y
        for dist in range(1, max_len + 1):
            x += dx
            y += dy
            if not (0 <= x < cols and 0 <= y < rows):
                if game.wrap_walls:
                    # Durchgangs-Modus: es gibt keine Wand, aussen weiterlaufen.
                    x %= cols
                    y %= rows
                else:
                    wall_feat[k] = 1.0 / dist
                    break
            cell = (x, y)
            if body_feat[k] == 0.0 and cell in occupied:
                body_feat[k] = 1.0 / dist
            if fruit_feat[k] == 0.0 and cell in fruits:
                fruit_feat[k] = 1.0 / dist

    # Gefahr direkt vor dem naechsten Zug (wie in der einfachen Wahrnehmung).
    candidate_cells = []
    for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
        mdx, mdy = relative_turn(direction, action).value
        candidate_cells.append((head_x + mdx, head_y + mdy))
    danger = [1.0 if flag else 0.0 for flag in game.danger_flags(candidate_cells)]

    # Frucht-Versatz, in die Sicht der Schlange gedreht und normiert.
    fruit_x, fruit_y = _nearest_fruit(game)
    ddx, ddy = fruit_x - head_x, fruit_y - head_y
    fruit_fwd = (ddx * fx + ddy * fy) / max_len
    fruit_right = (ddx * rx + ddy * ry) / max_len

    dir_onehot = [
        1.0 if direction == Direction.UP else 0.0,
        1.0 if direction == Direction.RIGHT else 0.0,
        1.0 if direction == Direction.DOWN else 0.0,
        1.0 if direction == Direction.LEFT else 0.0,
    ]

    # Wo liegt das Schwanzende, aus Sicht des Kopfes? Hilft der KI zu ahnen,
    # wie ihr eigener Koerper ungefaehr liegt (die Rays sehen nur die naechste
    # Koerperzelle, nicht den ganzen Verlauf).
    tail_x, tail_y = game.snake[-1]
    tdx, tdy = tail_x - head_x, tail_y - head_y
    t_fwd = tdx * fx + tdy * fy
    t_right = tdx * rx + tdy * ry
    tail_onehot = [
        1.0 if t_fwd > 0 and abs(t_fwd) >= abs(t_right) else 0.0,   # vorne
        1.0 if t_right > 0 and abs(t_right) > abs(t_fwd) else 0.0,  # rechts
        1.0 if t_fwd < 0 and abs(t_fwd) >= abs(t_right) else 0.0,   # hinten
        1.0 if t_right < 0 and abs(t_right) > abs(t_fwd) else 0.0,  # links
    ]

    length_norm = game.length / float(cols * rows)
    hunger = min(1.0, game.steps_since_fruit / 500.0)

    return np.array(
        wall_feat + body_feat + fruit_feat
        + danger
        + [fruit_fwd, fruit_right]
        + dir_onehot
        + tail_onehot
        + [length_norm, hunger],
        dtype=np.float32,
    )


# Register: Name -> (Funktion, Anzahl Eingangswerte). Der DQN-Trainer sucht sich
# ueber ai/dqn/config.py `perception` eine davon aus; die Neuroevolution benutzt
# unveraendert immer die einfache. So bleiben beide KIs unabhaengig, und wir
# koennen spaeter eine dritte Wahrnehmung ergaenzen, ohne irgendwo sonst etwas
# zu aendern.
PERCEPTIONS = {
    "simple": (perceive, INPUT_SIZE, FEATURE_LABELS),
    "rich": (perceive_rich, RICH_INPUT_SIZE, RICH_FEATURE_LABELS),
}


def get_perception(name: str):
    """(Funktion, Groesse, Bezeichnungen) fuer einen Wahrnehmungs-Namen."""
    if name not in PERCEPTIONS:
        raise ValueError(f"Unbekannte Wahrnehmung '{name}'. "
                         f"Moeglich: {', '.join(PERCEPTIONS)}")
    return PERCEPTIONS[name]


def describe(vector: np.ndarray) -> str:
    """Menschenlesbare Beschreibung eines Wahrnehmungsvektors (fuers Verstehen).

    Erkennt automatisch, ob es die einfache (11) oder die reiche (39) ist.
    """
    labels = RICH_FEATURE_LABELS if len(vector) == RICH_INPUT_SIZE else FEATURE_LABELS
    lines = [
        f"  {label:<22}: {value: .3f}"
        for label, value in zip(labels, vector)
    ]
    return "\n".join(lines)
