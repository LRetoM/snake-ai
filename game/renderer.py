"""Zeichnet den Spielzustand mit pygame -- die "Optik" des Spiels.

Der Renderer kennt die Spielregeln NICHT. Er bekommt nur ein fertiges SnakeGame
(den Zustand) und malt es huebsch auf den Bildschirm. Dadurch koennte man dasselbe
Spiel spaeter auch voellig anders darstellen (oder gar nicht -- fuer die KI).

Enthaelt:
- ein dezentes Schachbrett-Spielfeld
- die Schlange mit Farbverlauf (Kopf hell -> Schwanz dunkel) und Augen
- apfelartige Fruechte mit Glanzpunkt, Blatt und weichem Schimmer
- eine Kopfzeile mit Punkten / Laenge / Bestwert
- Overlays fuer Pause und Game Over
"""

from __future__ import annotations

import pygame

from .config import (
    CELL_SIZE,
    HEADER_HEIGHT,
    PLAY_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    Palette,
)
from .fonts import Font, load_font
from .snake_game import Direction, SnakeGame


def _lerp_color(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float):
    """Mischt zwei Farben. t=0 -> c1, t=1 -> c2, dazwischen linear interpoliert.

    Wird fuer den Farbverlauf der Schlange benutzt (Kopf -> Schwanz).
    """
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _best_font(size: int, bold: bool = False) -> Font:
    """Laedt eine Schrift in gewuenschter Groesse (siehe game/fonts.py, warum so)."""
    return load_font(size, bold=bold)


class Renderer:
    """Kapselt alles Zeichnen. Eine Instanz pro Fenster/Surface."""

    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface

        # Schriften einmalig laden (das ist vergleichsweise teuer).
        self.font_tiny = _best_font(16)
        self.font_small = _best_font(19)
        self.font_label = _best_font(15, bold=True)
        self.font_medium = _best_font(24, bold=True)
        self.font_score = _best_font(40, bold=True)
        self.font_large = _best_font(38, bold=True)
        self.font_title = _best_font(66, bold=True)

        # Statischen Feld-Hintergrund (Schachbrett) einmal vorzeichnen ->
        # spart Rechenzeit, weil er sich nie aendert.
        self._board_bg = self._build_board_background()

        # Weichen Schimmer fuer die Fruechte einmalig vorbereiten.
        self._fruit_glow = self._build_fruit_glow()

    # ------------------------------------------------------------------ #
    # Vorberechnete Flaechen
    # ------------------------------------------------------------------ #
    def _build_board_background(self) -> pygame.Surface:
        """Erzeugt das Schachbrettmuster des Spielfelds als eigene Flaeche."""
        bg = pygame.Surface((WINDOW_WIDTH, PLAY_HEIGHT))
        bg.fill(Palette.BOARD_A)
        cols = WINDOW_WIDTH // CELL_SIZE
        rows = PLAY_HEIGHT // CELL_SIZE
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
        # Mehrere konzentrische Kreise mit abnehmender Deckkraft = weicher Verlauf.
        for radius in range(center, 0, -1):
            alpha = int(70 * (1 - radius / center))
            color = (*Palette.FRUIT_GLOW, alpha)
            pygame.draw.circle(glow, color, (center, center), radius)
        return glow

    # ------------------------------------------------------------------ #
    # Koordinaten-Hilfen
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cell_rect(x: int, y: int) -> pygame.Rect:
        """Rechnet eine Gitterzelle (Spalte, Zeile) in ein Pixel-Rechteck um.

        Wichtig: Das Spielfeld liegt UNTER der Kopfzeile, daher + HEADER_HEIGHT.
        """
        return pygame.Rect(
            x * CELL_SIZE,
            HEADER_HEIGHT + y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )

    # ------------------------------------------------------------------ #
    # Hauptzeichen-Routine
    # ------------------------------------------------------------------ #
    def draw(self, game: SnakeGame, speed_name: str, high_score: int) -> None:
        """Zeichnet einen kompletten Frame des laufenden Spiels."""
        self.surface.fill(Palette.BG)
        self.surface.blit(self._board_bg, (0, HEADER_HEIGHT))
        self._draw_fruits(game)
        self._draw_snake(game)
        self._draw_header(game, speed_name, high_score)

    # ------------------------------------------------------------------ #
    # Fruechte
    # ------------------------------------------------------------------ #
    def _draw_fruits(self, game: SnakeGame) -> None:
        for (x, y) in game.fruits:
            rect = self._cell_rect(x, y)
            cx, cy = rect.center

            # 1) Weicher Schein hinter der Frucht.
            glow_rect = self._fruit_glow.get_rect(center=(cx, cy))
            self.surface.blit(self._fruit_glow, glow_rect)

            # 2) Der Apfel (roter Kreis).
            radius = CELL_SIZE // 2 - 4
            pygame.draw.circle(self.surface, Palette.FRUIT, (cx, cy), radius)

            # 3) Kleiner Glanzpunkt oben links -> wirkt runder/glaenzend.
            hl_r = max(2, radius // 3)
            pygame.draw.circle(
                self.surface,
                Palette.FRUIT_HIGHLIGHT,
                (cx - radius // 3, cy - radius // 3),
                hl_r,
            )

            # 4) Kleines Blatt oben.
            leaf = pygame.Rect(0, 0, radius, radius // 2)
            leaf.center = (cx + radius // 4, cy - radius)
            pygame.draw.ellipse(self.surface, Palette.FRUIT_LEAF, leaf)

    # ------------------------------------------------------------------ #
    # Schlange
    # ------------------------------------------------------------------ #
    def _draw_snake(self, game: SnakeGame) -> None:
        segments = game.snake
        n = len(segments)

        # Von hinten nach vorne zeichnen, damit der Kopf oben liegt.
        for i in range(n - 1, -1, -1):
            x, y = segments[i]
            rect = self._cell_rect(x, y)

            if i == 0:
                # Kopf: volle Zellengroesse, hellste Farbe.
                self._draw_head(game, rect)
            else:
                # Koerper: Farbverlauf ueber die Laenge + leichter Einzug,
                # damit die Segmente sichtbar getrennt wirken.
                t = (i - 1) / max(1, n - 2)  # 0 direkt hinter dem Kopf ... 1 am Schwanz
                color = _lerp_color(Palette.SNAKE_BODY, Palette.SNAKE_TAIL, t)
                inset = 2 if i < n - 1 else 4  # Schwanz etwas schmaler -> "spitzer"
                body = rect.inflate(-inset * 2, -inset * 2)
                pygame.draw.rect(self.surface, color, body, border_radius=9)

    def _draw_head(self, game: SnakeGame, rect: pygame.Rect) -> None:
        """Zeichnet den Kopf inkl. zweier Augen, die in Bewegungsrichtung schauen."""
        head = rect.inflate(-4, -4)
        pygame.draw.rect(self.surface, Palette.SNAKE_HEAD, head, border_radius=11)

        # Augenposition abhaengig von der Blickrichtung bestimmen.
        dx, dy = game.direction.value
        cx, cy = rect.center
        eye_r = max(2, CELL_SIZE // 9)
        offset = CELL_SIZE // 5  # Abstand der Augen von der Mitte

        if dx != 0:  # horizontal -> Augen uebereinander, nach vorne versetzt
            ex = cx + dx * offset
            eyes = [(ex, cy - offset), (ex, cy + offset)]
        else:        # vertikal -> Augen nebeneinander, nach vorne versetzt
            ey = cy + dy * offset
            eyes = [(cx - offset, ey), (cx + offset, ey)]

        for (ex, ey) in eyes:
            pygame.draw.circle(self.surface, Palette.SNAKE_EYE, (ex, ey), eye_r)

    # ------------------------------------------------------------------ #
    # Kopfzeile (Score / Laenge / Bestwert)
    # ------------------------------------------------------------------ #
    def _draw_header(self, game: SnakeGame, speed_name: str, high_score: int) -> None:
        # Hintergrund + feine Trennlinie zum Spielfeld.
        header_rect = pygame.Rect(0, 0, WINDOW_WIDTH, HEADER_HEIGHT)
        pygame.draw.rect(self.surface, Palette.HEADER_BG, header_rect)
        pygame.draw.line(
            self.surface, Palette.BORDER,
            (0, HEADER_HEIGHT - 1), (WINDOW_WIDTH, HEADER_HEIGHT - 1), 2,
        )

        # Drei Info-Bloecke gleichmaessig verteilt.
        third = WINDOW_WIDTH // 3
        self._draw_stat(third * 0 + third // 2, "PUNKTE", str(game.score), Palette.ACCENT)
        self._draw_stat(third * 1 + third // 2, "LAENGE", str(game.length), Palette.TEXT)
        self._draw_stat(third * 2 + third // 2, "BEST", str(high_score), Palette.TEXT_DIM)

        # Kleiner Geschwindigkeits-Hinweis oben links.
        speed_surf = self.font_tiny.render(f"Tempo: {speed_name}", True, Palette.TEXT_DIM)
        self.surface.blit(speed_surf, (14, 10))

    def _draw_stat(self, center_x: int, label: str, value: str, value_color) -> None:
        """Zeichnet einen Info-Block: kleines Label oben, grosse Zahl darunter."""
        label_surf = self.font_label.render(label, True, Palette.TEXT_DIM)
        value_surf = self.font_score.render(value, True, value_color)

        label_rect = label_surf.get_rect(center=(center_x, 34))
        value_rect = value_surf.get_rect(center=(center_x, 64))
        self.surface.blit(label_surf, label_rect)
        self.surface.blit(value_surf, value_rect)

    # ------------------------------------------------------------------ #
    # Overlays (halbtransparent ueber dem Spiel)
    # ------------------------------------------------------------------ #
    def _dim_screen(self, alpha: int = 190) -> None:
        """Legt einen halbtransparenten dunklen Schleier ueber das ganze Fenster."""
        veil = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        veil.fill((*Palette.OVERLAY, alpha))
        self.surface.blit(veil, (0, 0))

    def draw_pause_overlay(self) -> None:
        self._dim_screen(150)
        self._centered_text("PAUSE", self.font_title, Palette.TEXT, WINDOW_HEIGHT // 2 - 30)
        self._centered_text(
            "Leertaste = weiter    Esc = Menue",
            self.font_small, Palette.TEXT_DIM, WINDOW_HEIGHT // 2 + 30,
        )

    def draw_game_over_overlay(self, score: int, high_score: int, is_new_best: bool) -> None:
        self._dim_screen(200)
        cy = WINDOW_HEIGHT // 2

        self._centered_text("GAME OVER", self.font_title, Palette.ACCENT_WARN, cy - 90)
        self._centered_text(
            f"Punkte: {score}", self.font_large, Palette.TEXT, cy - 12,
        )

        if is_new_best:
            self._centered_text(
                "Neuer Bestwert!", self.font_medium, Palette.ACCENT, cy + 34,
            )
        else:
            self._centered_text(
                f"Bestwert: {high_score}", self.font_small, Palette.TEXT_DIM, cy + 34,
            )

        self._centered_text(
            "Enter = nochmal    Esc = Menue",
            self.font_small, Palette.TEXT_DIM, cy + 86,
        )

    def draw_win_overlay(self, score: int) -> None:
        self._dim_screen(200)
        cy = WINDOW_HEIGHT // 2
        self._centered_text("GEWONNEN!", self.font_title, Palette.ACCENT, cy - 70)
        self._centered_text(
            f"Feld gefuellt -- {score} Punkte", self.font_medium, Palette.TEXT, cy + 6,
        )
        self._centered_text(
            "Enter = nochmal    Esc = Menue",
            self.font_small, Palette.TEXT_DIM, cy + 60,
        )

    # ------------------------------------------------------------------ #
    # Startmenue / Einstellungen
    # ------------------------------------------------------------------ #
    def draw_menu(self, entries: list[tuple[str, str | None]], selected_index: int) -> None:
        """Zeichnet das Startmenue.

        entries ist eine Liste aus (Label, Wert). Ist Wert None, wird der Eintrag
        als grosser START-Knopf gezeichnet. selected_index markiert die aktive Zeile.
        """
        self.surface.fill(Palette.BG)

        # Titel + Untertitel.
        self._centered_text("SNAKE", self.font_title, Palette.ACCENT, 120)
        self._centered_text(
            "Klassisch  ·  Pfeiltasten oder WASD",
            self.font_small, Palette.TEXT_DIM, 170,
        )

        # Zeilenblock vertikal etwa mittig anordnen.
        panel_x = 60
        panel_w = WINDOW_WIDTH - 2 * panel_x
        row_h = 56
        gap = 12
        start_y = 250

        for i, (label, value) in enumerate(entries):
            y = start_y + i * (row_h + gap)
            selected = (i == selected_index)

            if value is None:
                self._draw_menu_button(label, panel_w, y, row_h, selected)
            else:
                self._draw_menu_row(label, value, panel_x, panel_w, y, row_h, selected)

        # Steuerungshinweis unten.
        self._centered_text(
            "Pfeil hoch/runter waehlen   ·   links/rechts aendern   ·   Enter start",
            self.font_tiny, Palette.TEXT_DIM, WINDOW_HEIGHT - 34,
        )

    def _draw_menu_row(self, label, value, panel_x, panel_w, y, row_h, selected):
        """Eine Einstellungszeile: Label links, Wert rechts (mit Pfeilen wenn aktiv)."""
        rect = pygame.Rect(panel_x, y, panel_w, row_h)
        if selected:
            pygame.draw.rect(self.surface, (28, 40, 36), rect, border_radius=12)
            pygame.draw.rect(self.surface, Palette.ACCENT, rect, width=2, border_radius=12)

        label_color = Palette.TEXT if selected else Palette.TEXT_DIM
        label_surf = self.font_medium.render(label, True, label_color)
        self.surface.blit(
            label_surf, label_surf.get_rect(midleft=(panel_x + 24, rect.centery))
        )

        value_text = f"<   {value}   >" if selected else value
        value_color = Palette.ACCENT if selected else Palette.TEXT
        value_surf = self.font_medium.render(value_text, True, value_color)
        self.surface.blit(
            value_surf, value_surf.get_rect(midright=(panel_x + panel_w - 24, rect.centery))
        )

    def _draw_menu_button(self, label, panel_w, y, row_h, selected):
        """Der START-Knopf, mittig und deutlich hervorgehoben."""
        btn_w = 240
        rect = pygame.Rect(0, y, btn_w, row_h)
        rect.centerx = WINDOW_WIDTH // 2

        bg = Palette.ACCENT if selected else (32, 37, 52)
        pygame.draw.rect(self.surface, bg, rect, border_radius=14)
        if not selected:
            pygame.draw.rect(self.surface, Palette.BORDER, rect, width=2, border_radius=14)

        text_color = Palette.BG if selected else Palette.TEXT
        text_surf = self.font_medium.render(label, True, text_color)
        self.surface.blit(text_surf, text_surf.get_rect(center=rect.center))

    # ------------------------------------------------------------------ #
    # kleiner Text-Helfer
    # ------------------------------------------------------------------ #
    def _centered_text(self, text: str, font: Font, color, y: int) -> None:
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(WINDOW_WIDTH // 2, y))
        self.surface.blit(surf, rect)
