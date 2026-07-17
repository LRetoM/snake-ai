"""Zeichnet den Spielzustand mit pygame -- die "Optik" des Spiels.

Der Renderer kennt die Spielregeln NICHT. Er bekommt nur ein fertiges SnakeGame
(den Zustand) und malt es huebsch. Dadurch koennte man dasselbe Spiel spaeter
voellig anders darstellen (oder gar nicht -- fuer das schnelle KI-Training).

Optik-Merkmale:
- fluessige Bewegung durch Interpolation zwischen zwei Spielschritten
  (die Schlange GLEITET, statt von Zelle zu Zelle zu springen)
- weiche, anti-aliaste Kanten (gfxdraw) fuer Aepfel und Schlangenkoerper
- die Schlange als zusammenhaengende Roehre mit Farbverlauf und Augen
- moderne Schrift (Poppins) und ein Text-Cache fuer gute Performance
"""

from __future__ import annotations

import pygame
import pygame.gfxdraw as gfxdraw

from .config import (
    CELL_SIZE,
    HEADER_HEIGHT,
    PLAY_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    Palette,
)
from .fonts import Font, load_font
from .snake_game import SnakeGame

# Breite des Schlangenkoerpers relativ zur Zelle (laesst einen kleinen Rand frei).
BODY_WIDTH = int(CELL_SIZE * 0.76)
BODY_RADIUS = BODY_WIDTH // 2


def _lerp_color(c1, c2, t: float):
    """Mischt zwei Farben. t=0 -> c1, t=1 -> c2."""
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _fill_circle(surface: pygame.Surface, x: float, y: float, r: float, color) -> None:
    """Gefuellter Kreis mit weicher (anti-aliaster) Kante."""
    if r < 1:
        return
    xi, yi, ri = int(x), int(y), int(r)
    gfxdraw.filled_circle(surface, xi, yi, ri, color)
    gfxdraw.aacircle(surface, xi, yi, ri, color)


class Renderer:
    """Kapselt alles Zeichnen. Eine Instanz pro Fenster/Surface."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface

        # Schriften einmalig laden (verschiedene Groessen/Schnitte).
        self.f_title = load_font(64, "semibold")
        self.f_h2 = load_font(30, "semibold")
        self.f_stat = load_font(34, "semibold")
        self.f_body = load_font(21, "medium")
        self.f_body_reg = load_font(19, "regular")
        self.f_label = load_font(13, "semibold")
        self.f_small = load_font(16, "regular")
        self.f_tiny = load_font(14, "regular")

        # Statischen Feld-Hintergrund (Schachbrett) einmal vorzeichnen.
        self._board_bg = self._build_board_background()
        # Weichen Schimmer fuer die Fruechte einmalig vorbereiten.
        self._fruit_glow = self._build_fruit_glow()
        # Wiederverwendbarer dunkler Schleier fuer Overlays.
        self._veil = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)

        # Kleiner Cache fuer gerenderten Text (spart Rechenzeit pro Frame).
        self._text_cache: dict[tuple, pygame.Surface] = {}

    # ------------------------------------------------------------------ #
    # Vorberechnete Flaechen
    # ------------------------------------------------------------------ #
    def _build_board_background(self) -> pygame.Surface:
        """Erzeugt das dezente Schachbrettmuster des Spielfelds."""
        bg = pygame.Surface((WINDOW_WIDTH, PLAY_HEIGHT))
        bg.fill(Palette.BOARD_A)
        cols = WINDOW_WIDTH // CELL_SIZE + 1
        rows = PLAY_HEIGHT // CELL_SIZE + 1
        for x in range(cols):
            for y in range(rows):
                if (x + y) % 2 == 1:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(bg, Palette.BOARD_B, rect)
        return bg

    def _build_fruit_glow(self) -> pygame.Surface:
        """Weicher, halbtransparenter Schein hinter jeder Frucht."""
        size = CELL_SIZE * 2
        glow = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        for radius in range(center, 0, -1):
            alpha = int(60 * (1 - radius / center) ** 2)
            gfxdraw.filled_circle(
                glow, center, center, radius, (*Palette.FRUIT_GLOW, alpha)
            )
        return glow

    # ------------------------------------------------------------------ #
    # Text-Cache
    # ------------------------------------------------------------------ #
    def _text(self, font: Font, text: str, color) -> pygame.Surface:
        """Rendert Text (gecacht), damit gleiche Texte nicht jeden Frame neu entstehen."""
        key = (id(font), text, color)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 400:
                self._text_cache.clear()
            surf = font.render(text, color)
            self._text_cache[key] = surf
        return surf

    def _blit_centered(self, font, text, color, cx, cy) -> None:
        surf = self._text(font, text, color)
        self.surface.blit(surf, surf.get_rect(center=(cx, cy)))

    # ------------------------------------------------------------------ #
    # Koordinaten-Hilfe
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cell_center(x: float, y: float) -> tuple[float, float]:
        """Pixel-Mittelpunkt einer (evtl. gebrochenen) Gitterposition."""
        px = x * CELL_SIZE + CELL_SIZE / 2
        py = HEADER_HEIGHT + y * CELL_SIZE + CELL_SIZE / 2
        return px, py

    # ------------------------------------------------------------------ #
    # Hauptzeichen-Routine
    # ------------------------------------------------------------------ #
    def draw(self, game, speed_name, high_score, prev_snake=None, alpha=1.0) -> None:
        """Zeichnet einen kompletten Frame des laufenden Spiels.

        prev_snake/alpha steuern die fluessige Bewegung: alpha=0 zeigt die
        Schlange an ihrer vorherigen Position, alpha=1 an der aktuellen; dazwischen
        gleitet sie. Sind keine Werte gegeben, wird ohne Interpolation gezeichnet.
        """
        self.surface.fill(Palette.BG)
        self.surface.blit(self._board_bg, (0, HEADER_HEIGHT))
        self._draw_fruits(game)
        self._draw_snake(game, prev_snake, alpha)
        self._draw_header(game, speed_name, high_score)

    # ------------------------------------------------------------------ #
    # Fruechte (Aepfel)
    # ------------------------------------------------------------------ #
    def _draw_fruits(self, game) -> None:
        radius = CELL_SIZE // 2 - 4
        for (x, y) in game.fruits:
            cx, cy = self._cell_center(x, y)
            cx, cy = int(cx), int(cy)

            # 1) weicher Schein
            self.surface.blit(self._fruit_glow, self._fruit_glow.get_rect(center=(cx, cy)))
            # 2) Apfel (weiche Kante)
            _fill_circle(self.surface, cx, cy, radius, Palette.FRUIT)
            # 3) Glanzpunkt
            _fill_circle(self.surface, cx - radius // 3, cy - radius // 3,
                         max(2, radius // 3), Palette.FRUIT_HIGHLIGHT)
            # 4) kleines Blatt
            leaf = pygame.Rect(0, 0, radius, radius // 2)
            leaf.center = (cx + radius // 4, cy - radius + 1)
            pygame.draw.ellipse(self.surface, Palette.FRUIT_LEAF, leaf)

    # ------------------------------------------------------------------ #
    # Schlange (fluessige, zusammenhaengende Roehre)
    # ------------------------------------------------------------------ #
    def _draw_snake(self, game, prev_snake, alpha) -> None:
        curr = game.snake
        n = len(curr)
        centers = self._interp_centers(prev_snake, curr, alpha)

        # Von hinten (Schwanz) nach vorne (Kopf) zeichnen -> Kopf liegt oben.
        for i in range(n - 1, -1, -1):
            color = self._snake_color(i, n)
            cx, cy = centers[i]

            # Verbindungsstueck zum vorderen Nachbarn fuellt die Luecke -> Roehre.
            if i > 0:
                px, py = centers[i - 1]
                if abs(px - cx) <= 1.5 * CELL_SIZE and abs(py - cy) <= 1.5 * CELL_SIZE:
                    pygame.draw.line(self.surface, color, (px, py), (cx, cy), BODY_WIDTH)

            _fill_circle(self.surface, cx, cy, BODY_RADIUS, color)

        # Augen zuletzt oben auf den Kopf.
        if n:
            self._draw_eyes(game, centers[0])

    def _interp_centers(self, prev_snake, curr, alpha):
        """Berechnet fuer jedes aktuelle Segment den (interpolierten) Pixel-Mittelpunkt."""
        centers = []
        prev = prev_snake or []
        lp = len(prev)
        for i, (cx, cy) in enumerate(curr):
            # Wo war dieses Segment im vorigen Schritt? (neu gewachsene bleiben stehen)
            fx, fy = prev[i] if i < lp else (cx, cy)
            # Bei Wand-Durchgang springt ein Segment ueber den ganzen Bildschirm ->
            # dann NICHT interpolieren, sonst wuerde es quer durchs Feld gleiten.
            if abs(cx - fx) > 1 or abs(cy - fy) > 1:
                ix, iy = cx, cy
            else:
                ix = fx + (cx - fx) * alpha
                iy = fy + (cy - fy) * alpha
            centers.append(self._cell_center(ix, iy))
        return centers

    @staticmethod
    def _snake_color(i: int, n: int):
        """Farbverlauf: Kopf hell, Koerper mittel, Schwanz dunkel."""
        if i == 0:
            return Palette.SNAKE_HEAD
        t = (i - 1) / max(1, n - 2)
        return _lerp_color(Palette.SNAKE_BODY, Palette.SNAKE_TAIL, t)

    def _draw_eyes(self, game, head_center) -> None:
        """Zwei Augen, die in Blickrichtung schauen (mit kleinem Glanzpunkt)."""
        cx, cy = head_center
        dx, dy = game.direction.value
        # senkrecht zur Blickrichtung (fuer den seitlichen Augenabstand)
        sx, sy = -dy, dx
        fwd = CELL_SIZE * 0.14
        side = CELL_SIZE * 0.17
        eye_r = max(2.0, CELL_SIZE * 0.11)

        for s in (+1, -1):
            ex = cx + dx * fwd + sx * side * s
            ey = cy + dy * fwd + sy * side * s
            _fill_circle(self.surface, ex, ey, eye_r, Palette.SNAKE_EYE)
            _fill_circle(self.surface, ex + dx * 1.2, ey + dy * 1.2,
                         max(1.0, eye_r * 0.4), (235, 240, 245))

    # ------------------------------------------------------------------ #
    # Kopfzeile (Score / Laenge / Bestwert)
    # ------------------------------------------------------------------ #
    def _draw_header(self, game, speed_name, high_score) -> None:
        pygame.draw.rect(self.surface, Palette.HEADER_BG,
                         pygame.Rect(0, 0, WINDOW_WIDTH, HEADER_HEIGHT))
        pygame.draw.line(self.surface, Palette.BORDER,
                         (0, HEADER_HEIGHT - 1), (WINDOW_WIDTH, HEADER_HEIGHT - 1), 2)

        third = WINDOW_WIDTH // 3
        self._draw_stat(third * 0 + third // 2, "PUNKTE", str(game.score), Palette.ACCENT)
        self._draw_stat(third * 1 + third // 2, "LÄNGE", str(game.length), Palette.TEXT)
        self._draw_stat(third * 2 + third // 2, "BEST", str(high_score), Palette.TEXT_DIM)

    def _draw_stat(self, center_x, label, value, value_color) -> None:
        self._blit_centered(self.f_label, label, Palette.TEXT_DIM, center_x, 34)
        self._blit_centered(self.f_stat, value, value_color, center_x, 63)

    # ------------------------------------------------------------------ #
    # Overlays
    # ------------------------------------------------------------------ #
    def _dim_screen(self, alpha: int = 190) -> None:
        self._veil.fill((*Palette.OVERLAY, alpha))
        self.surface.blit(self._veil, (0, 0))

    def draw_pause_overlay(self) -> None:
        self._dim_screen(150)
        cx = WINDOW_WIDTH // 2
        self._blit_centered(self.f_title, "Pause", Palette.TEXT, cx, WINDOW_HEIGHT // 2 - 26)
        self._blit_centered(self.f_small, "Leertaste = weiter      Esc = Menü",
                            Palette.TEXT_DIM, cx, WINDOW_HEIGHT // 2 + 34)

    def draw_game_over_overlay(self, score, high_score, is_new_best) -> None:
        self._dim_screen(205)
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        self._blit_centered(self.f_title, "Game Over", Palette.ACCENT_WARN, cx, cy - 88)
        self._blit_centered(self.f_h2, f"{score} Punkte", Palette.TEXT, cx, cy - 20)
        if is_new_best:
            self._blit_centered(self.f_body, "Neuer Bestwert!", Palette.ACCENT, cx, cy + 30)
        else:
            self._blit_centered(self.f_small, f"Bestwert: {high_score}",
                                Palette.TEXT_DIM, cx, cy + 28)
        self._blit_centered(self.f_small, "Enter = nochmal      Esc = Menü",
                            Palette.TEXT_DIM, cx, cy + 80)

    def draw_win_overlay(self, score) -> None:
        self._dim_screen(205)
        cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        self._blit_centered(self.f_title, "Gewonnen!", Palette.ACCENT, cx, cy - 60)
        self._blit_centered(self.f_h2, f"Feld gefüllt — {score} Punkte",
                            Palette.TEXT, cx, cy + 6)
        self._blit_centered(self.f_small, "Enter = nochmal      Esc = Menü",
                            Palette.TEXT_DIM, cx, cy + 56)

    # ------------------------------------------------------------------ #
    # Startmenue / Einstellungen
    # ------------------------------------------------------------------ #
    def draw_menu(self, entries: list[tuple[str, str | None]], selected_index: int) -> None:
        self.surface.fill(Palette.BG)
        cx = WINDOW_WIDTH // 2

        self._blit_centered(self.f_title, "SNAKE", Palette.ACCENT, cx, 118)
        self._blit_centered(self.f_small, "Klassisch  ·  Pfeiltasten oder WASD",
                            Palette.TEXT_DIM, cx, 168)

        panel_x = 64
        panel_w = WINDOW_WIDTH - 2 * panel_x
        row_h = 54
        gap = 14
        start_y = 250

        for i, (label, value) in enumerate(entries):
            y = start_y + i * (row_h + gap)
            selected = (i == selected_index)
            if value is None:
                self._draw_menu_button(label, y, row_h, selected)
            else:
                self._draw_menu_row(label, value, panel_x, panel_w, y, row_h, selected)

        self._blit_centered(
            self.f_tiny,
            "Hoch / Runter  wählen      ·      Links / Rechts  ändern      ·      Enter  Start",
            Palette.TEXT_DIM, cx, WINDOW_HEIGHT - 36,
        )

    def _draw_menu_row(self, label, value, panel_x, panel_w, y, row_h, selected) -> None:
        rect = pygame.Rect(panel_x, y, panel_w, row_h)
        if selected:
            pygame.draw.rect(self.surface, (26, 38, 34), rect, border_radius=14)
            pygame.draw.rect(self.surface, Palette.ACCENT, rect, width=2, border_radius=14)

        label_color = Palette.TEXT if selected else Palette.TEXT_DIM
        label_surf = self._text(self.f_body, label, label_color)
        self.surface.blit(label_surf, label_surf.get_rect(midleft=(panel_x + 26, rect.centery)))

        value_color = Palette.ACCENT if selected else Palette.TEXT
        value_surf = self._text(self.f_body, value, value_color)
        vx = panel_x + panel_w - 26
        self.surface.blit(value_surf, value_surf.get_rect(midright=(vx, rect.centery)))
        if selected:
            # dezente Pfeile links/rechts vom Wert
            arrow_l = self._text(self.f_body, "‹", Palette.ACCENT)
            arrow_r = self._text(self.f_body, "›", Palette.ACCENT)
            vw = value_surf.get_width()
            self.surface.blit(arrow_l, arrow_l.get_rect(midright=(vx - vw - 14, rect.centery)))
            self.surface.blit(arrow_r, arrow_r.get_rect(midleft=(vx + 12, rect.centery)))

    def _draw_menu_button(self, label, y, row_h, selected) -> None:
        rect = pygame.Rect(0, y, 240, row_h)
        rect.centerx = WINDOW_WIDTH // 2
        bg = Palette.ACCENT if selected else (30, 35, 50)
        pygame.draw.rect(self.surface, bg, rect, border_radius=16)
        if not selected:
            pygame.draw.rect(self.surface, Palette.BORDER, rect, width=2, border_radius=16)
        text_color = Palette.BG if selected else Palette.TEXT
        self._blit_centered(self.f_h2, label, text_color, rect.centerx, rect.centery)
