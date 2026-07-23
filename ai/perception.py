"""Wahrnehmung: verwandelt den Spielzustand in einen Zahlen-Vektor fuer die KI.

Das ist die "Sinnesorgan"-Schicht. Die KI bekommt NIE den Code oder eine
fertige Loesung -- sie bekommt nur Zahlen, die beschreiben, was gerade "zu
sehen" ist. Mal ein kleiner Ausschnitt (8 Tast-Strahlen, ein Fenster um den
Kopf), mal ("full_board") das komplette Feld -- genau wie ein Mensch beim
Spielen ja auch den GANZEN Bildschirm sieht, nicht nur einen Ausschnitt davon.
Voll sehen zu duerfen ist also erlaubt; die Grenze verlaeuft nicht bei "wie
viel sieht sie", sondern bei "rechnet der CODE ihr etwas vor" (siehe unten).

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


# =========================================================================== #
# NAHBEREICHS-KARTE ("rich_gridN") -- gegen das Selbst-Einsperr-Plateau
# =========================================================================== #
# Warum reicht "rich" (die 8 Tast-Strahlen) allein nicht?
#
# Jeder Strahl meldet nur die NAECHSTE Koerperzelle in einer geraden Linie.
# Legt sich der Koerper aber wie eine Spirale um den Kopf, "sieht" keiner der
# 8 Strahlen die Windungen ZWISCHEN sich -- die KI merkt die Falle oft erst,
# wenn sie schon fast drin sitzt. Genau das zeigen die Zahlen im Dashboard:
# bei langer Schlange sind >50% aller Tode Selbstkollisionen, obwohl die
# Pruefung laengst nicht mehr steigt (Plateau).
#
# Die Nahbereichs-Karte gibt der KI echtes "peripheres Sehen" statt nur
# Antennen in 8 Linien: ein window x window Ausschnitt der naeheren
# Umgebung, egozentrisch gedreht (vorne = oben, wie beim Rest der reichen
# Wahrnehmung). Jede Zelle bekommt nur EINEN Sinneswert (leer/eigener
# Koerper/Frucht/Wand) -- nach wie vor reine Wahrnehmung. KEIN Flood-Fill,
# KEINE berechnete "wie viel Platz habe ich noch"-Kennzahl -- das waere
# eine ausgerechnete Heuristik, die der KI die Loesung des Einsperr-
# Problems vorwegnaehme. Sie muss selbst lernen, welche Muster gefaehrlich
# sind, wir geben ihr nur schaerfere Augen.
#
# Groesser = mehr Vorwarnzeit vor entfernteren Spiralen, aber mehr Eingaenge
# (window**2 zusaetzliche Werte) -> groesseres Netz, langsameres Training.
# Deshalb mehrere Groessen registriert (5/7/9) statt nur einer festen Wahl --
# das laesst sich im Menue/config.py direkt gegeneinander austesten.

def _local_grid(game: SnakeGame, window: int) -> list[float]:
    """window x window Zellen um den Kopf, egozentrisch (vorne = oben).

    Zeilen laufen von ganz vorne zu ganz hinten, Spalten von ganz links zu
    ganz rechts (aus Sicht der Schlange) -- dieselbe Zeilenweise-Reihenfolge
    wie man ein Blatt Papier liest. Werte: 0.0 = leer, 1.0 = eigener Koerper,
    0.5 = Frucht, -1.0 = Wand (nur ohne Durchgangs-Modus; die Kopfzelle selbst
    ist immer 0.0, sie enthaelt keine Information).
    """
    half = window // 2
    head_x, head_y = game.head
    fx, fy = game.direction.value
    rx, ry = -fy, fx

    occupied = game.occupied
    fruits = game.fruits
    cols, rows = game.cols, game.rows

    cells: list[float] = []
    for fwd in range(half, -half - 1, -1):
        for right in range(-half, half + 1):
            if fwd == 0 and right == 0:
                cells.append(0.0)
                continue
            x = head_x + fwd * fx + right * rx
            y = head_y + fwd * fy + right * ry
            if not (0 <= x < cols and 0 <= y < rows):
                if game.wrap_walls:
                    x %= cols
                    y %= rows
                else:
                    cells.append(-1.0)
                    continue
            cell = (x, y)
            if cell in occupied:
                cells.append(1.0)
            elif cell in fruits:
                cells.append(0.5)
            else:
                cells.append(0.0)
    return cells


def make_rich_grid_perception(window: int):
    """Baut eine Wahrnehmungsfunktion: die reichen 39 Werte + eine window x
    window Nahbereichs-Karte (siehe `_local_grid`). Gibt (Funktion,
    Eingangsgroesse, Klartext-Labels) zurueck -- passend zum PERCEPTIONS-Format.
    """
    half = window // 2
    grid_labels = [
        f"Nah vorne{fwd:+d}/rechts{right:+d}"
        for fwd in range(half, -half - 1, -1)
        for right in range(-half, half + 1)
    ]
    labels = list(RICH_FEATURE_LABELS) + grid_labels
    input_size = RICH_INPUT_SIZE + window * window

    def perceive_rich_grid(game: SnakeGame) -> np.ndarray:
        base = perceive_rich(game)
        grid = np.array(_local_grid(game, window), dtype=np.float32)
        return np.concatenate([base, grid])

    return perceive_rich_grid, input_size, labels


# =========================================================================== #
# VOLLE FELD-SICHT ("full_board")
# =========================================================================== #
# Die konsequente Fortsetzung der Nahbereichs-Karte: statt eines Ausschnitts
# rund um den Kopf sieht die KI hier das GESAMTE Spielfeld auf einen Blick --
# wie ein Mensch, der beim Spielen ja auch den ganzen Bildschirm vor sich hat,
# nicht nur ein Guckloch. Jede Feldzelle bekommt genau EINEN Sinneswert (leer/
# eigener Koerper/Frucht) -- weiterhin reine Wahrnehmung, kein Flood-Fill,
# keine berechnete Kennzahl "wie sicher ist dieser Weg". Ob und wie sie daraus
# lernt, sich selbst nicht mehr einzusperren, muss sie weiterhin durch
# Erfahrung selbst herausfinden -- wir oeffnen ihr nur die Augen, wir spielen
# nicht fuer sie.
#
# Anders als bei den Nahbereichs-Fenstern (egozentrisch gedreht, "vorne oben")
# bleibt das volle Feld in FESTER Ausrichtung (so wie es auf dem Bildschirm
# liegt) -- eine Drehung eines ganzen Rechtecks um einen beliebigen Kopfpunkt
# wuerde nicht mehr sauber ins Raster passen. Deshalb bekommt die KI hier
# zusaetzlich ihre absolute Kopfposition UND Blickrichtung mit (bei den
# kleinen Fenstern unnoetig, weil "vorne" durch die Drehung schon feststand).
# Fruchtrichtung/Schwanzrichtung (wie bei "rich") sind hier ueberfluessig --
# beides liegt ja bereits sichtbar im Feld selbst.
#
# Die Eingangsgroesse haengt hier von der BRETTGROESSE ab (11 + cols*rows) --
# anders als bei den anderen Wahrnehmungen laesst sie sich also nicht einmalig
# beim Modul-Import festlegen (das Brett ist waehlbar, siehe Menue
# "Brettgroesse" / DQNConfig.grid_cols/grid_rows). Groesse+Labels kommen
# deshalb erst zur Laufzeit aus `make_full_board_perception(cols, rows)` --
# die Wahrnehmungsfunktion selbst braucht dafuer KEINE Anpassung, sie liest
# Breite/Hoehe ohnehin live aus `game.cols`/`game.rows`.
def make_full_board_perception(cols: int, rows: int):
    """(Funktion, Eingangsgroesse, Labels) fuer die volle Feld-Sicht auf
    einem cols x rows Brett -- passend zum PERCEPTIONS-Format."""
    size = 11 + cols * rows
    labels = (
        ["Gefahr geradeaus", "Gefahr rechts", "Gefahr links",
         "Richtung: oben", "Richtung: rechts", "Richtung: unten", "Richtung: links",
         "Kopf x (normiert)", "Kopf y (normiert)", "Laenge", "Hunger"]
        + [f"Feld ({x},{y})" for y in range(rows) for x in range(cols)]
    )
    return perceive_full_board, size, labels


def perceive_full_board(game: SnakeGame) -> np.ndarray:
    """Volle Feld-Sicht: 11 Basiswerte + eine Zelle pro Feldposition.

    Aufbau:
      0-2   Gefahr geradeaus / rechts / links (wie ueberall sonst)
      3-6   aktuelle Blickrichtung (one-hot, absolut)
      7-8   Kopfposition x/y, normiert auf 0..1
      9     Laenge der Schlange (0..1, anteilig am Spielfeld)
      10    Hunger: Schritte seit der letzten Frucht (0..1, bei ~500 gedeckelt)
      11+   das ganze Feld, zeilenweise (y dann x): 0.0 leer, 1.0 eigener
            Koerper, 0.5 Frucht
    """
    head_x, head_y = game.head
    direction = game.direction
    cols, rows = game.cols, game.rows

    candidate_cells = []
    for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
        dx, dy = relative_turn(direction, action).value
        candidate_cells.append((head_x + dx, head_y + dy))
    danger = [1.0 if flag else 0.0 for flag in game.danger_flags(candidate_cells)]

    dir_onehot = [
        1.0 if direction == Direction.UP else 0.0,
        1.0 if direction == Direction.RIGHT else 0.0,
        1.0 if direction == Direction.DOWN else 0.0,
        1.0 if direction == Direction.LEFT else 0.0,
    ]

    length_norm = game.length / float(cols * rows)
    hunger = min(1.0, game.steps_since_fruit / 500.0)

    board = [0.0] * (cols * rows)
    for (x, y) in game.occupied:
        board[y * cols + x] = 1.0
    for (x, y) in game.fruits:
        board[y * cols + x] = 0.5

    return np.array(
        danger + dir_onehot
        + [head_x / cols, head_y / rows, length_norm, hunger]
        + board,
        dtype=np.float32,
    )


# Register: Name -> (Funktion, Anzahl Eingangswerte, Labels). Der DQN-Trainer
# sucht sich ueber ai/dqn/config.py `perception` eine davon aus; die
# Neuroevolution benutzt unveraendert immer die einfache. So bleiben beide KIs
# unabhaengig, und wir koennen jederzeit eine weitere Wahrnehmung ergaenzen,
# ohne irgendwo sonst etwas zu aendern.
PERCEPTIONS = {
    "simple": (perceive, INPUT_SIZE, FEATURE_LABELS),
    "rich": (perceive_rich, RICH_INPUT_SIZE, RICH_FEATURE_LABELS),
}
for _window in (5, 7, 9):
    PERCEPTIONS[f"rich_grid{_window}"] = make_rich_grid_perception(_window)
del _window


# =========================================================================== #
# SYMMETRIE-SPIEGELUNG (links-rechts) -- gratis doppelte Trainingsdaten
# =========================================================================== #
# Snake ist links-rechts spiegelsymmetrisch: jede egozentrische Erfahrung gilt
# GESPIEGELT genauso (was rechts von der Schlange lag, liegt im Spiegelbild
# links, und ein Rechts-Abbiegen im Original ist ein Links-Abbiegen im
# Spiegelbild). Fuers Lernen heisst das: jede gezogene Erinnerung kann mit
# 50% Wahrscheinlichkeit gespiegelt "wiederverwendet" werden, ohne dass die
# Schlange dafuer einen einzigen Zug mehr spielen muss -- effektiv doppelte
# Trainingsdaten geschenkt. Reine Trainings-Mechanik (ai/dqn/memory.py), an
# der Wahrnehmung selbst aendert sich nichts.
#
# Fuer jede Wahrnehmung braucht es dafuer eine PERMUTATION + VORZEICHEN-MASKE:
# gespiegelter_vektor = vektor[perm] * sign. Das funktioniert nur, weil alle
# hier gespiegelten Wahrnehmungen EGOZENTRISCH sind (vorne/rechts/hinten/links
# relativ zur Schlange) -- "full_board" ist dagegen ABSOLUT ausgerichtet und
# enthaelt eine normierte Kopfposition, die sich nicht per simpler Permutation+
# Vorzeichen spiegeln laesst (dafuer braeuchte es eine Verschiebung, kein
# reines Umsortieren) -- deshalb bewusst NICHT in MIRROR_MAPS, solange diese
# Wahrnehmung nicht aktiv im Einsatz ist.
def _rich_mirror() -> tuple[np.ndarray, np.ndarray]:
    """Spiegel-Permutation + Vorzeichen fuer die 39 "rich"-Werte.

    Jeder 8er-Block (Wand/Koerper/Frucht in 8 Richtungen, siehe _RAY_OFFSETS)
    wird um die Vorne-Hinten-Achse gespiegelt: [vorne, vorne-rechts, rechts,
    hinten-rechts, hinten, hinten-links, links, vorne-links] wird zu [vorne,
    vorne-links, links, hinten-links, hinten, hinten-rechts, rechts,
    vorne-rechts] -- also Position k <- Original-Position (8-k) mod 8, kurz
    [0,7,6,5,4,3,2,1]. Gefahr/Blickrichtung/Schwanz: rechts<->links tauschen,
    geradeaus/vorne/hinten/oben/unten bleiben. Der Frucht-Rechts-Versatz
    (Index 28) ist ein VORZEICHENBEHAFTETER Wert (nicht 0/1) -- beim Spiegeln
    kehrt sich sein Vorzeichen um, statt nur die Position zu wechseln.
    """
    ray8 = [0, 7, 6, 5, 4, 3, 2, 1]
    perm = (
        ray8                                        # Wand      (0-7)
        + [8 + k for k in ray8]                      # Koerper   (8-15)
        + [16 + k for k in ray8]                      # Frucht    (16-23)
        + [24, 26, 25]                                # Gefahr: geradeaus/rechts/links
        + [27, 28]                                    # Frucht-Versatz vorne/rechts
        + [29, 32, 31, 30]                            # Richtung: oben/rechts/unten/links
        + [33, 36, 35, 34]                            # Schwanz: vorne/rechts/hinten/links
        + [37, 38]                                    # Laenge, Hunger
    )
    sign = np.ones(RICH_INPUT_SIZE, dtype=np.float32)
    sign[28] = -1.0   # Frucht-Rechts-Versatz kehrt sich beim Spiegeln um
    return np.array(perm, dtype=np.int64), sign


def _simple_mirror() -> tuple[np.ndarray, np.ndarray]:
    """Spiegel-Permutation fuer die 11 "simple"-Werte (siehe FEATURE_LABELS):
    Gefahr rechts<->links, Richtung rechts<->links, Frucht links<->rechts;
    geradeaus/oben/unten bleiben. Alles hier sind 0/1-Flaggen -> kein
    Vorzeichenwechsel noetig, reine Permutation."""
    perm = [0, 2, 1, 3, 6, 5, 4, 8, 7, 9, 10]
    return np.array(perm, dtype=np.int64), np.ones(INPUT_SIZE, dtype=np.float32)


def _grid_mirror_local(window: int) -> np.ndarray:
    """Spiegel-Permutation NUR fuer die window x window Nahbereichs-Karte
    (lokale Indizes 0..window*window-1, siehe _local_grid): jede Zeile
    (fester Vorne-Abstand) wird links-rechts umgedreht, die Zeilen-
    Reihenfolge (vorne->hinten) bleibt."""
    local = []
    for r in range(window):
        row_start = r * window
        for j in range(window):
            local.append(row_start + (window - 1 - j))
    return np.array(local, dtype=np.int64)


def _build_mirror_maps() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rich_perm, rich_sign = _rich_mirror()
    simple_perm, simple_sign = _simple_mirror()
    maps = {
        "simple": (simple_perm, simple_sign),
        "rich": (rich_perm, rich_sign),
    }
    for window in (5, 7, 9):
        grid_perm = np.concatenate([rich_perm, RICH_INPUT_SIZE + _grid_mirror_local(window)])
        grid_sign = np.concatenate([rich_sign, np.ones(window * window, dtype=np.float32)])
        maps[f"rich_grid{window}"] = (grid_perm, grid_sign)
    return maps


MIRROR_MAPS: dict[str, tuple[np.ndarray, np.ndarray]] = _build_mirror_maps()

# STRAIGHT (0) bleibt, LEFT (1) <-> RIGHT (2) tauschen -- unabhaengig von der
# Wahrnehmung, weil das Action-Enum fuer alle gleich ist.
ACTION_MIRROR = np.array([0, 2, 1], dtype=np.int64)


def mirror_perception(name: str, vector: np.ndarray) -> np.ndarray | None:
    """Spiegelt einen Wahrnehmungsvektor links-rechts, falls fuer `name` eine
    Spiegelung definiert ist -- sonst None (aktuell nur "full_board" ohne,
    siehe Kommentar oben bei MIRROR_MAPS)."""
    entry = MIRROR_MAPS.get(name)
    if entry is None:
        return None
    perm, sign = entry
    return vector[perm] * sign


# Wahrnehmungen, deren Eingangsgroesse von der Brettgroesse abhaengt (aktuell
# nur "full_board") -- lassen sich nicht wie oben einmalig beim Modul-Import
# vorberechnen. get_perception() braucht fuer sie zusaetzlich cols/rows.
BOARD_DEPENDENT_PERCEPTIONS = {
    "full_board": make_full_board_perception,
}


def get_perception(name: str, cols: int | None = None, rows: int | None = None):
    """(Funktion, Groesse, Bezeichnungen) fuer einen Wahrnehmungs-Namen.

    Brettgroessen-abhaengige Wahrnehmungen (aktuell "full_board") brauchen
    zusaetzlich `cols`/`rows` -- ohne die laesst sich ihre Eingangsgroesse
    nicht bestimmen (das Netz muesste sonst raten, wie viele Feldzellen
    ueberhaupt kommen). Aufrufer: `ai/dqn/trainer.py` uebergibt dafuer
    `cfg.grid_cols`/`cfg.grid_rows`; `watch_ai.py` die Brettgroesse aus dem
    geladenen Champion-Checkpoint.
    """
    if name in PERCEPTIONS:
        return PERCEPTIONS[name]
    if name in BOARD_DEPENDENT_PERCEPTIONS:
        if cols is None or rows is None:
            raise ValueError(
                f"Wahrnehmung '{name}' haengt von der Brettgroesse ab -- "
                "get_perception() braucht dafuer cols und rows."
            )
        return BOARD_DEPENDENT_PERCEPTIONS[name](cols, rows)
    bekannt = list(PERCEPTIONS) + list(BOARD_DEPENDENT_PERCEPTIONS)
    raise ValueError(f"Unbekannte Wahrnehmung '{name}'. Moeglich: {', '.join(bekannt)}")


def describe(vector: np.ndarray) -> str:
    """Menschenlesbare Beschreibung eines Wahrnehmungsvektors (fuers Verstehen).

    Erkennt die passenden Labels automatisch anhand der Vektorlaenge -- ueber
    alle registrierten Wahrnehmungen (auch die Nahbereichs-Varianten), statt
    nur "einfach" und "reich" fest zu unterscheiden.
    """
    for _fn, size, labels in PERCEPTIONS.values():
        if size == len(vector):
            return "\n".join(
                f"  {label:<22}: {value: .3f}" for label, value in zip(labels, vector)
            )
    return "\n".join(f"  [{i}]: {v: .3f}" for i, v in enumerate(vector))
