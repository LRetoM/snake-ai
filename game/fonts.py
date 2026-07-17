"""Kleiner Font-Wrapper, der einen pygame-Fehler unter Python 3.14 umgeht.

Hintergrund (fuer die Neugierigen):
In dieser pygame-Installation fehlt das SDL_ttf-basierte 'font'-Modul, deshalb
greift der freetype-Fallback. Der wiederum hat unter Python 3.14 einen zirkulaeren
Import (pygame/font.py <-> pygame/sysfont.py), sodass sowohl `pygame.font` als auch
`pygame.freetype` beim Import abstuerzen. Die zugrunde liegende C-Extension
`pygame._freetype` laedt jedoch einwandfrei.

Loesung: Wir bauen einen minimalen Ersatz auf Basis von `pygame._freetype`, der
genau die render()-Signatur bietet, die der Renderer erwartet -- ganz ohne das
defekte `sysfont`. So bleibt die vorhandene Umgebung unveraendert und das Spiel
laeuft trotzdem. Sollte pygame spaeter aktualisiert werden, kann man hier problemlos
wieder auf pygame.font umstellen.
"""

from __future__ import annotations

import os

import pygame
import pygame._freetype as _freetype

# pygame bringt diese Schrift immer mit -> keine Systemsuche noetig
# (die Systemsuche liefe ueber das defekte pygame.sysfont).
_DEFAULT_TTF = os.path.join(os.path.dirname(pygame.__file__), "freesansbold.ttf")


def _ensure_init() -> None:
    """Stellt sicher, dass das freetype-Modul initialisiert ist (idempotent)."""
    if not _freetype.get_init():
        _freetype.init()


class Font:
    """Minimaler Ersatz fuer pygame.font.Font mit gleicher render()-Signatur."""

    def __init__(self, size: int, bold: bool = False) -> None:
        _ensure_init()
        self._font = _freetype.Font(_DEFAULT_TTF, size)
        self._font.antialiased = True
        self._font.pad = True   # gleichmaessige Zeilenhoehe unabhaengig vom Text
        if bold:
            self._font.strong = True     # kuenstlich fetten
            self._font.strength = 0.10

    def render(self, text: str, antialias: bool, color) -> pygame.Surface:
        """Wie pygame.font.Font.render(): gibt eine fertige Surface zurueck."""
        self._font.antialiased = antialias
        surface, _rect = self._font.render(text, color)
        return surface

    def size(self, text: str) -> tuple[int, int]:
        """Breite/Hoehe des Textes in Pixeln (wie pygame.font.Font.size())."""
        return self._font.get_rect(text).size


def load_font(size: int, bold: bool = False) -> Font:
    """Bequemer Konstruktor -- vom Renderer benutzt."""
    return Font(size, bold=bold)
