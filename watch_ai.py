"""KI zuschauen: laesst den gespeicherten Neuroevolution-Champion im normalen
Spielfenster spielen -- exakt dieselbe Spiel-Engine (game/snake_game.py) und
derselbe Renderer (game/renderer.py) wie beim Menschen-Spiel (play_human.py).

Der einzige Unterschied zu play_human.py: Statt der Tastatur waehlt ein
neuronales Netz die Zuege. Die KI hat dabei -- genau wie ein Mensch -- nur die
Wahrnehmung (ai/perception.py) als Eingabe und "drueckt" nur geradeaus/links/
rechts. So siehst du ehrlich, wie gut die trainierte KI wirklich spielt.

Start:
    source venv/bin/activate      (Mac)   /   venv\\Scripts\\activate   (Windows)
    python watch_ai.py

Voraussetzung: mindestens einmal trainiert (python train_evolution.py), damit
models/evo_champion.pt existiert. Ohne das gibt es eine klare Fehlermeldung
statt eines Absturzes.
"""

from __future__ import annotations

import os
import sys

import pygame
import torch

from game.config import (
    DEFAULT_SPEED_INDEX,
    SPEED_PRESETS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    GameConfig,
)
from game.renderer import Renderer
from game.snake_game import Action, SnakeGame
from ai.network import SnakeNet
from ai.perception import perceive

FPS = 60
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "evo_champion.pt")

# Wichtig: Das SPIEL selbst kennt KEINE "Verhungern"-Regel -- das ist bewusst so
# (Leitplanke: die KI-Trainingsregeln duerfen keine Spielregeln sein). Beim
# Training bricht der Trainer lange Partien extern ab (steps_since_fruit).
# Beim Zuschauen tun wir dasselbe, nur grosszuegiger: eine mittelmaessige KI
# kann sonst in eine SICHERE Endlosschleife geraten (nie Wand, nie sich selbst,
# aber auch nie eine Frucht) und wuerde ohne dieses Limit einfach ewig weiter-
# laufen -- fuer den Zuschauer nicht von einem eingefrorenen Programm zu
# unterscheiden. Das ist keine Spielregel, sondern reiner Zuschau-Komfort.
STUCK_LIMIT = 400


def load_champion(path: str = _MODEL_PATH) -> tuple[SnakeNet, dict]:
    """Laedt den gespeicherten Champion-Checkpoint (Netz + Trainings-Metadaten).

    Bricht mit einer klaren, verstaendlichen Meldung ab, falls noch nie
    trainiert wurde -- statt mit einem kryptischen Absturz.
    """
    if not os.path.exists(path):
        print(
            "Kein trainierter Champion gefunden.\n"
            f"  Erwartet unter: {path}\n\n"
            "Starte zuerst ein Training, z.B.:\n"
            "  python train_evolution.py\n"
            "(im Dashboard mindestens ein paar Generationen laufen lassen,\n"
            "der beste Bot wird automatisch gespeichert)."
        )
        sys.exit(1)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = SnakeNet(hidden=checkpoint["hidden"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()  # nur Inferenz -- kein Training, keine Gradienten noetig
    return net, checkpoint


class WatchApp:
    """Wie play_human.App, aber die KI (nicht die Tastatur) waehlt die Zuege."""

    MENU = "MENU"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    GAME_OVER = "GAME_OVER"
    WIN = "WIN"

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Snake — KI zuschauen")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        self.net, self.checkpoint = load_champion()

        # Einzige Einstellung im Menue: Anzeige-Geschwindigkeit. Fruchtanzahl/
        # Wandmodus kommen bewusst FEST aus dem Checkpoint -- so wird die KI
        # fair unter genau den Bedingungen getestet, unter denen sie trainiert
        # wurde (ein mit 3 Fruechten gezuechteter Bot soll nicht ploetzlich
        # mit 1 Frucht laufen, das waere kein ehrlicher Vergleich).
        self.speed_index = DEFAULT_SPEED_INDEX
        self.menu_row = 0

        self.game: SnakeGame | None = None
        self.move_interval_ms = 1000.0
        self.move_accumulator = 0.0
        self.prev_snake: list = []

        self.state = WatchApp.MENU
        self.running = True
        self.session_best = 0
        self.episode_new_best = False
        self.episode_stuck = False  # True, wenn die Partie am Verhungern-Limit endete

    # ================================================================== #
    # Hauptschleife
    # ================================================================== #
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(FPS)
            self._handle_events()
            if self.state == WatchApp.PLAYING:
                self._update_play(dt)
            self._draw()
            pygame.display.flip()
        pygame.quit()

    # ================================================================== #
    # Eingabe (die KI steuert die Schlange -- der Mensch steuert nur App-Zustaende)
    # ================================================================== #
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if self.state == WatchApp.MENU:
                    self._on_menu_key(event.key)
                elif self.state == WatchApp.PLAYING:
                    self._on_play_key(event.key)
                elif self.state == WatchApp.PAUSED:
                    self._on_pause_key(event.key)
                else:
                    self._on_over_key(event.key)

    def _on_menu_key(self, key: int) -> None:
        row_count = 2  # Geschwindigkeit + START
        if key in (pygame.K_UP, pygame.K_w):
            self.menu_row = (self.menu_row - 1) % row_count
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu_row = (self.menu_row + 1) % row_count
        elif key in (pygame.K_LEFT, pygame.K_a) and self.menu_row == 0:
            self.speed_index = (self.speed_index - 1) % len(SPEED_PRESETS)
        elif key in (pygame.K_RIGHT, pygame.K_d) and self.menu_row == 0:
            self.speed_index = (self.speed_index + 1) % len(SPEED_PRESETS)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_episode()
        elif key == pygame.K_ESCAPE:
            self.running = False

    def _on_play_key(self, key: int) -> None:
        # Keine Richtungstasten hier -- die KI steuert. Nur Pause moeglich.
        if key in (pygame.K_SPACE, pygame.K_p, pygame.K_ESCAPE):
            self.state = WatchApp.PAUSED

    def _on_pause_key(self, key: int) -> None:
        if key in (pygame.K_SPACE, pygame.K_p, pygame.K_RETURN, pygame.K_KP_ENTER):
            self.state = WatchApp.PLAYING
        elif key == pygame.K_ESCAPE:
            self.state = WatchApp.MENU

    def _on_over_key(self, key: int) -> None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_episode()  # neue Partie, dasselbe Netz
        elif key == pygame.K_ESCAPE:
            self.state = WatchApp.MENU

    # ================================================================== #
    # Ablauf
    # ================================================================== #
    def _start_episode(self) -> None:
        """Setzt eine frische Partie unter den Trainings-Bedingungen des Champions auf."""
        config = GameConfig(
            grid_cols=self.checkpoint.get("grid_cols", 20),
            grid_rows=self.checkpoint.get("grid_rows", 20),
            fruit_count=self.checkpoint.get("fruit_count", 1),
            wrap_walls=self.checkpoint.get("wrap_walls", False),
        )
        self.game = SnakeGame(config)

        moves_per_second = SPEED_PRESETS[self.speed_index].moves_per_second
        self.move_interval_ms = 1000.0 / moves_per_second
        self.move_accumulator = 0.0
        self.prev_snake = list(self.game.snake)
        self.episode_new_best = False
        self.episode_stuck = False
        self.state = WatchApp.PLAYING

    def _update_play(self, dt: int) -> None:
        """Fixed-Timestep-Schleife wie in play_human.py -- nur entscheidet hier
        das Netz statt der Tastatur, welche Aktion als naechstes drankommt."""
        assert self.game is not None
        self.move_accumulator += dt

        while self.move_accumulator >= self.move_interval_ms:
            self.move_accumulator -= self.move_interval_ms
            self.prev_snake = list(self.game.snake)

            # Die KI "sieht" nur die 11 Wahrnehmungszahlen -- wie ein Mensch nur
            # den Bildschirm sieht, nicht den Code.
            observation = perceive(self.game)
            with torch.no_grad():
                output = self.net(torch.from_numpy(observation))
            action = Action(int(torch.argmax(output).item()))

            result = self.game.step_action(action)

            if result.won:
                self._finish_episode(result.score)
                self.state = WatchApp.WIN
                return
            if not result.alive:
                self._finish_episode(result.score)
                self.state = WatchApp.GAME_OVER
                return
            if self.game.steps_since_fruit >= STUCK_LIMIT:
                # Kein Kollisionstod -- die KI haengt nur in einer sicheren
                # Endlosschleife fest. Siehe Kommentar bei STUCK_LIMIT oben.
                self.episode_stuck = True
                self._finish_episode(result.score)
                self.state = WatchApp.GAME_OVER
                return

    def _finish_episode(self, score: int) -> None:
        self.episode_new_best = score > self.session_best
        self.session_best = max(self.session_best, score)

    # ================================================================== #
    # Zeichnen
    # ================================================================== #
    def _draw(self) -> None:
        if self.state == WatchApp.MENU:
            self._draw_menu()
            return

        assert self.game is not None
        speed_name = SPEED_PRESETS[self.speed_index].name

        if self.state in (WatchApp.PLAYING, WatchApp.PAUSED):
            alpha = min(1.0, self.move_accumulator / self.move_interval_ms)
        else:
            alpha = 1.0
        self.renderer.draw(self.game, speed_name, self.session_best, self.prev_snake, alpha)

        if self.state == WatchApp.PAUSED:
            self.renderer.draw_pause_overlay()
        elif self.state == WatchApp.GAME_OVER:
            title = "Kein Fortschritt" if self.episode_stuck else "Game Over"
            self.renderer.draw_game_over_overlay(
                self.game.score, self.session_best, self.episode_new_best, title=title
            )
        elif self.state == WatchApp.WIN:
            self.renderer.draw_win_overlay(self.game.score)

    def _draw_menu(self) -> None:
        cp = self.checkpoint
        wall_txt = "Durchgang" if cp.get("wrap_walls", False) else "Tödlich"
        subtitle2 = (
            f"Champion: {cp['score']} Punkte (Generation {cp['generation']})  ·  "
            f"trainiert mit {cp.get('fruit_count', 1)} Frucht(en), Wand: {wall_txt}"
        )
        entries = [
            ("Geschwindigkeit", SPEED_PRESETS[self.speed_index].name),
            ("ZUSCHAUEN STARTEN", None),
        ]
        self.renderer.draw_menu(
            entries, self.menu_row,
            title="KI ZUSCHAUEN",
            subtitle="Neuroevolution-Champion spielt selbst",
            subtitle2=subtitle2,
        )


def main() -> None:
    WatchApp().run()


if __name__ == "__main__":
    main()
