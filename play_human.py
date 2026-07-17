"""Snake -- von Hand spielbar mit Pfeiltasten (oder WASD).

Startet das Fenster, zeigt ein Einstellungsmenue (Geschwindigkeit, Anzahl
Fruechte, Wand-Modus) und laesst dich anschliessend klassisch Snake spielen.

Start:
    source venv/bin/activate
    python play_human.py

Diese Datei ist der "Dirigent": Sie holt Eingaben, taktet die Zeit und ruft die
Spiellogik (game/snake_game.py) sowie das Zeichnen (game/renderer.py) auf.
Sie enthaelt selbst KEINE Spielregeln -- die liegen ausschliesslich im Spiel.
"""

from __future__ import annotations

import json
import os

import pygame

from game.config import (
    DEFAULT_FRUIT_INDEX,
    DEFAULT_SPEED_INDEX,
    DEFAULT_WALL_INDEX,
    FRUIT_COUNT_OPTIONS,
    SPEED_PRESETS,
    WALL_MODES,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    GameConfig,
)
from game.renderer import Renderer
from game.snake_game import Direction, SnakeGame

FPS = 60  # Zeichen-/Eingabe-Takt (unabhaengig von der Spielgeschwindigkeit)

# Wo der Bestwert gespeichert wird (pro Modus). logs/ wird nicht committed.
HIGHSCORE_FILE = os.path.join(os.path.dirname(__file__), "logs", "highscore.json")

# Tastenzuordnung: Pfeiltasten UND WASD steuern die vier Richtungen.
DIRECTION_KEYS = {
    pygame.K_UP: Direction.UP,
    pygame.K_w: Direction.UP,
    pygame.K_DOWN: Direction.DOWN,
    pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT,
    pygame.K_a: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_d: Direction.RIGHT,
}


class App:
    """Haelt den Gesamtzustand (Menue vs. Spiel) und die Hauptschleife zusammen."""

    # Moegliche Zustaende des Programms.
    MENU = "MENU"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Snake")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        # Aktuelle Menue-Auswahl (Indizes in die Optionslisten aus config.py).
        self.speed_index = DEFAULT_SPEED_INDEX
        self.fruit_index = DEFAULT_FRUIT_INDEX
        self.wall_index = DEFAULT_WALL_INDEX
        self.menu_row = 0  # welche Menuezeile gerade markiert ist

        # Laufendes Spiel + Zeitsteuerung.
        self.game: SnakeGame | None = None
        self.move_interval_ms = 1000.0  # wird beim Start gesetzt
        self.move_accumulator = 0.0     # gesammelte Zeit fuer den naechsten Schritt
        # Schlangen-Zustand VOR dem letzten Schritt -> fuer fluessige Interpolation.
        self.prev_snake: list = []

        self.state = App.MENU
        self.is_new_best = False
        self.running = True

        self.high_scores: dict[str, int] = self._load_high_scores()

    # ================================================================== #
    # Hauptschleife
    # ================================================================== #
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS)  # Millisekunden seit letztem Frame
            self._handle_events()
            if self.state == App.PLAYING:
                self._update_play(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()

    # ================================================================== #
    # Eingabe
    # ================================================================== #
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.state == App.MENU:
                    self._on_menu_key(event.key)
                elif self.state == App.PLAYING:
                    self._on_play_key(event.key)
                elif self.state == App.PAUSED:
                    self._on_pause_key(event.key)
                else:  # GAME_OVER oder WIN
                    self._on_over_key(event.key)

    def _on_menu_key(self, key: int) -> None:
        row_count = 4  # 3 Einstellungen + START
        if key in (pygame.K_UP, pygame.K_w):
            self.menu_row = (self.menu_row - 1) % row_count
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu_row = (self.menu_row + 1) % row_count
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._change_setting(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._change_setting(+1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_game()
        elif key == pygame.K_ESCAPE:
            self.running = False

    def _change_setting(self, delta: int) -> None:
        """Aendert die aktuell markierte Einstellung (mit Umlauf)."""
        if self.menu_row == 0:
            self.speed_index = (self.speed_index + delta) % len(SPEED_PRESETS)
        elif self.menu_row == 1:
            self.fruit_index = (self.fruit_index + delta) % len(FRUIT_COUNT_OPTIONS)
        elif self.menu_row == 2:
            self.wall_index = (self.wall_index + delta) % len(WALL_MODES)
        # Zeile 3 ist der START-Knopf -> keine Einstellung.

    def _on_play_key(self, key: int) -> None:
        if key in DIRECTION_KEYS:
            assert self.game is not None
            self.game.change_direction(DIRECTION_KEYS[key])
        elif key in (pygame.K_SPACE, pygame.K_p, pygame.K_ESCAPE):
            self.state = App.PAUSED

    def _on_pause_key(self, key: int) -> None:
        if key in (pygame.K_SPACE, pygame.K_p, pygame.K_RETURN, pygame.K_KP_ENTER):
            self.state = App.PLAYING
        elif key == pygame.K_ESCAPE:
            self.state = App.MENU

    def _on_over_key(self, key: int) -> None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_game()  # gleiche Einstellungen, neue Runde
        elif key == pygame.K_ESCAPE:
            self.state = App.MENU

    # ================================================================== #
    # Spielablauf
    # ================================================================== #
    def _start_game(self) -> None:
        """Baut aus der Menue-Auswahl eine Runde und wechselt ins Spiel."""
        config = GameConfig(
            fruit_count=FRUIT_COUNT_OPTIONS[self.fruit_index],
            wrap_walls=(self.wall_index == 1),
        )
        self.game = SnakeGame(config)

        moves_per_second = SPEED_PRESETS[self.speed_index].moves_per_second
        self.move_interval_ms = 1000.0 / moves_per_second
        self.move_accumulator = 0.0
        self.prev_snake = list(self.game.snake)

        self.is_new_best = False
        self.state = App.PLAYING

    def _update_play(self, dt: int) -> None:
        """Bewegt die Schlange im festen Takt weiter (fixed timestep).

        Warum so? Das Fenster laeuft mit 60 FPS fuer fluessige Eingabe, aber die
        Schlange soll nur alle move_interval_ms EINEN Schritt machen. Wir sammeln
        die vergangene Zeit auf und machen so viele Schritte, wie "faellig" sind.
        Vorteil: Die Geschwindigkeit ist unabhaengig von der Bildrate -- auf jedem
        Rechner gleich schnell.
        """
        assert self.game is not None
        self.move_accumulator += dt

        while self.move_accumulator >= self.move_interval_ms:
            self.move_accumulator -= self.move_interval_ms
            self.prev_snake = list(self.game.snake)  # Zustand vor dem Schritt merken
            result = self.game.step()

            if result.won:
                self._register_score(result.score)
                self.state = App.WIN
                return
            if not result.alive:
                self._register_score(result.score)
                self.state = App.GAME_OVER
                return

    # ================================================================== #
    # Zeichnen
    # ================================================================== #
    def _draw(self) -> None:
        if self.state == App.MENU:
            self.renderer.draw_menu(self._menu_entries(), self.menu_row)
            return

        assert self.game is not None
        speed_name = SPEED_PRESETS[self.speed_index].name
        high = self._current_high_score()

        # Gleit-Fortschritt zwischen zwei Schritten (0..1) fuer fluessige Bewegung.
        if self.state in (App.PLAYING, App.PAUSED):
            alpha = min(1.0, self.move_accumulator / self.move_interval_ms)
        else:
            alpha = 1.0  # bei Game Over / Sieg steht die Schlange still
        self.renderer.draw(self.game, speed_name, high, self.prev_snake, alpha)

        if self.state == App.PAUSED:
            self.renderer.draw_pause_overlay()
        elif self.state == App.GAME_OVER:
            self.renderer.draw_game_over_overlay(self.game.score, high, self.is_new_best)
        elif self.state == App.WIN:
            self.renderer.draw_win_overlay(self.game.score)

    def _menu_entries(self) -> list[tuple[str, str | None]]:
        """Baut die Zeilen des Menues aus der aktuellen Auswahl."""
        return [
            ("Geschwindigkeit", SPEED_PRESETS[self.speed_index].name),
            ("Früchte", str(FRUIT_COUNT_OPTIONS[self.fruit_index])),
            ("Wände", WALL_MODES[self.wall_index]),
            ("START", None),
        ]

    # ================================================================== #
    # Bestwerte (pro Modus gespeichert)
    # ================================================================== #
    def _mode_key(self, fruit_count: int, wrap: bool) -> str:
        return f"{fruit_count}f_{'durchgang' if wrap else 'wand'}"

    def _current_mode_key(self) -> str:
        if self.game is not None:
            return self._mode_key(self.game.fruit_count, self.game.wrap_walls)
        return self._mode_key(
            FRUIT_COUNT_OPTIONS[self.fruit_index], self.wall_index == 1
        )

    def _current_high_score(self) -> int:
        return self.high_scores.get(self._current_mode_key(), 0)

    def _register_score(self, score: int) -> None:
        """Prueft, ob ein neuer Bestwert erreicht wurde, und speichert ihn."""
        key = self._current_mode_key()
        if score > self.high_scores.get(key, 0):
            self.high_scores[key] = score
            self.is_new_best = True
            self._save_high_scores()

    def _load_high_scores(self) -> dict[str, int]:
        try:
            with open(HIGHSCORE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # nur saubere int-Werte uebernehmen
            return {str(k): int(v) for k, v in data.items()}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def _save_high_scores(self) -> None:
        try:
            os.makedirs(os.path.dirname(HIGHSCORE_FILE), exist_ok=True)
            with open(HIGHSCORE_FILE, "w", encoding="utf-8") as fh:
                json.dump(self.high_scores, fh, indent=2)
        except OSError:
            pass  # Bestwert nicht speichern zu koennen soll das Spiel nicht abbrechen


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
