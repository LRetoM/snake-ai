"""Zentrale Einstellungen, Farben und Layout-Werte fuer das Snake-Spiel.

Alles, was man "einstellen" kann, wird hier gebuendelt. Der Vorteil: Wenn du
z.B. eine neue Geschwindigkeit oder ein anderes Farbschema willst, aenderst du
nur diese Datei -- die Spiellogik und das Zeichnen bleiben unberuehrt.

Hinweis fuer Python-Einsteiger:
- Ein `@dataclass` ist eine bequeme Art, einen "Datencontainer" zu bauen.
  Man beschreibt nur die Felder, und Python erzeugt Konstruktor & Co. automatisch.
- `Tuple[int, int, int]` ist ein RGB-Farbwert, z.B. (74, 222, 128) = ein Gruenton.
"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Spielfeld-Layout (in Rasterzellen und Pixeln)
# --------------------------------------------------------------------------- #
# Das Spielfeld ist ein Gitter aus GRID_COLS x GRID_ROWS Zellen.
# Jede Zelle ist CELL_SIZE Pixel gross. Oben gibt es eine Kopfzeile (Header)
# fuer Punktestand & Infos.
GRID_COLS = 20            # Anzahl Spalten
GRID_ROWS = 20            # Anzahl Zeilen
CELL_SIZE = 34            # Kantenlaenge einer Zelle in Pixeln
HEADER_HEIGHT = 92        # Hoehe der oberen Infoleiste in Pixeln

PLAY_WIDTH = GRID_COLS * CELL_SIZE          # Breite des Spielfelds
PLAY_HEIGHT = GRID_ROWS * CELL_SIZE         # Hoehe des Spielfelds
WINDOW_WIDTH = PLAY_WIDTH                    # Fensterbreite
WINDOW_HEIGHT = PLAY_HEIGHT + HEADER_HEIGHT  # Fensterhoehe (Feld + Header)


# --------------------------------------------------------------------------- #
# Geschwindigkeits-Presets
# --------------------------------------------------------------------------- #
# "moves_per_second" = wie viele Schritte die Schlange pro Sekunde macht.
# Das Fenster wird trotzdem fluessig mit 60 FPS gezeichnet -- die Spielgeschwindig-
# keit ist davon entkoppelt (siehe fixed-timestep-Schleife in play_human.py).
@dataclass(frozen=True)
class SpeedPreset:
    name: str               # Anzeigename im Menue
    moves_per_second: float  # Schritte pro Sekunde


SPEED_PRESETS: list[SpeedPreset] = [
    SpeedPreset("Gemütlich", 6),
    SpeedPreset("Normal", 9),
    SpeedPreset("Schnell", 13),
    SpeedPreset("Rasant", 18),
    SpeedPreset("Wahnsinn", 26),
]
DEFAULT_SPEED_INDEX = 1  # "Normal"


# Auswahlmoeglichkeiten fuer die Anzahl gleichzeitig sichtbarer Fruechte (1 bis 10).
FRUIT_COUNT_OPTIONS: list[int] = list(range(1, 11))
DEFAULT_FRUIT_INDEX = 0  # 1 Frucht


# Wand-Modus: klassisch (Wand = Tod) oder Durchgang (Schlange kommt auf der
# gegenueberliegenden Seite wieder heraus -- bekannter Snake-Variantenmodus).
WALL_MODES: list[str] = ["Tödlich", "Durchgang"]
DEFAULT_WALL_INDEX = 0


@dataclass
class GameConfig:
    """Konkrete Einstellungen fuer EINE Spielrunde.

    Wird aus den Menue-Auswahlen zusammengebaut und an das Spiel uebergeben.
    """
    grid_cols: int = GRID_COLS
    grid_rows: int = GRID_ROWS
    fruit_count: int = 1          # Anzahl gleichzeitig liegender Fruechte
    wrap_walls: bool = False      # True = Durchgang statt Tod an der Wand


# --------------------------------------------------------------------------- #
# Farbpalette (modernes, dunkles Design)
# --------------------------------------------------------------------------- #
# RGB-Werte. Zentral gesammelt, damit das Spiel ein einheitliches Aussehen hat.
class Palette:
    # Hintergruende
    BG = (16, 18, 27)              # Fenster-Hintergrund (sehr dunkles Blaugrau)
    BOARD_A = (24, 27, 38)         # Schachbrettmuster Feld -- Ton A
    BOARD_B = (28, 32, 45)         # Schachbrettmuster Feld -- Ton B
    HEADER_BG = (13, 15, 22)       # Kopfzeile
    BORDER = (44, 49, 66)          # feine Trennlinien / Rahmen

    # Schlange (Farbverlauf von Kopf -> Schwanz)
    SNAKE_HEAD = (74, 222, 128)    # helles Gruen
    SNAKE_BODY = (34, 197, 94)     # mittleres Gruen
    SNAKE_TAIL = (21, 128, 61)     # dunkles Gruen
    SNAKE_EYE = (16, 18, 27)       # Augen (dunkel)

    # Frucht (Apfel)
    FRUIT = (239, 68, 68)          # Rot
    FRUIT_HIGHLIGHT = (252, 165, 165)  # Glanzpunkt
    FRUIT_LEAF = (34, 197, 94)     # Blatt
    FRUIT_GLOW = (239, 68, 68)     # Schimmer um die Frucht

    # Text
    TEXT = (229, 231, 235)         # heller Standardtext
    TEXT_DIM = (148, 163, 184)     # gedaempfter Text (Labels)
    ACCENT = (74, 222, 128)        # Akzentfarbe (Gruen)
    ACCENT_WARN = (248, 113, 113)  # Warn-/Game-Over-Rot

    # Overlays
    OVERLAY = (10, 12, 18)         # halbtransparent darueber (mit Alpha im Code)
