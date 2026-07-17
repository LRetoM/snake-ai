"""Schrift-Laden fuer das Spiel -- mit sauberen Schnitten und pygame-3.14-Fix.

Zwei Dinge passieren hier:

1) Wir laden eine mitgelieferte, moderne Schrift (Poppins, freie OFL-Lizenz,
   liegt in assets/fonts/). Dadurch sieht die Oberflaeche auf JEDEM Rechner
   gleich aus (Mac wie Windows) und wir koennen die Strichstaerke ueber echte
   Schnitte steuern (Regular/Medium/SemiBold) statt kuenstlich zu "verfetten".

2) Wir umgehen einen pygame-Bug unter Python 3.14: Das normale Text-Modul
   (pygame.font / pygame.freetype) stuerzt beim Import ab (zirkulaerer Import in
   pygame.sysfont). Die C-Extension pygame._freetype laedt aber sauber -- die
   nutzen wir direkt. Deshalb hier bewusst KEIN "import pygame.font".

Faellt die Poppins-Datei mal weg, greifen wir automatisch auf die von pygame
mitgelieferte Schrift zurueck, damit nie etwas abstuerzt.
"""

from __future__ import annotations

import os

import pygame
import pygame._freetype as _freetype

# Ordner mit den mitgelieferten Schriftdateien.
_ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")

# Verfuegbare Schnitte (Strichstaerken) -> Dateiname.
_WEIGHT_FILES = {
    "regular": "Poppins-Regular.ttf",
    "medium": "Poppins-Medium.ttf",
    "semibold": "Poppins-SemiBold.ttf",
}

# Notfall-Schrift, die pygame immer mitbringt.
_FALLBACK_TTF = os.path.join(os.path.dirname(pygame.__file__), "freesansbold.ttf")


def _ensure_init() -> None:
    """Stellt sicher, dass das freetype-Modul initialisiert ist (idempotent)."""
    if not _freetype.get_init():
        _freetype.init()


def _resolve_path(weight: str) -> str:
    """Findet die Schriftdatei fuer einen Schnitt, sonst die Fallback-Schrift."""
    filename = _WEIGHT_FILES.get(weight, _WEIGHT_FILES["regular"])
    path = os.path.join(_ASSET_DIR, filename)
    return path if os.path.exists(path) else _FALLBACK_TTF


class Font:
    """Duennes Wrapper um pygame._freetype.Font mit einfacher render()-API."""

    def __init__(self, size: int, weight: str = "regular") -> None:
        _ensure_init()
        self._font = _freetype.Font(_resolve_path(weight), size)
        self._font.antialiased = True   # weiche Kanten
        self._font.pad = True           # gleichmaessige Zeilenhoehe
        # Etwas Laufweite macht Ueberschriften ruhiger/edler.
        self._font.kerning = True

    def render(self, text: str, color) -> pygame.Surface:
        """Gibt eine fertige Surface mit dem gerenderten Text zurueck."""
        surface, _rect = self._font.render(text, color)
        return surface

    def size(self, text: str) -> tuple[int, int]:
        """Breite/Hoehe des Textes in Pixeln."""
        return self._font.get_rect(text).size


def load_font(size: int, weight: str = "regular") -> Font:
    """Bequemer Konstruktor -- vom Renderer benutzt."""
    return Font(size, weight=weight)
