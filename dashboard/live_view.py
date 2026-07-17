"""Live-Dashboard fuer das Neuroevolution-Training.

Zeigt in Echtzeit:
- ein RASTER vieler Schlangen der aktuellen Generation, die gleichzeitig spielen
- ein STATISTIK-Panel (Generation, Score, Fitness, Todesursachen, Diversität)
- eine LERNKURVE (Score-Entwicklung ueber die Generationen)

Steuerung waehrend des Trainings:
- Leertaste : Pause / weiter
- T         : Turbo an/aus (ohne Anzeige rechnen -> maximale Geschwindigkeit)
- Pfeil hoch/runter : Anzeige-Geschwindigkeit der Schlangen
- Esc       : zurueck ins Menue

Das Dashboard "treibt" den Trainer (ai/evolution/train_evolution.py) Schritt fuer
Schritt an und zeichnet dazwischen -- die Rechenlogik selbst enthaelt kein pygame.
"""

from __future__ import annotations

import math
import time

import pygame

from game.config import Palette
from game.fonts import load_font
from ai.network import DEFAULT_HIDDEN
from ai.evolution.train_evolution import EvolutionConfig, EvolutionTrainer, GenerationStats

# ----------------------------- Fenster-Layout ------------------------------ #
WIN_W, WIN_H = 1180, 720
HEADER_H = 60
PAD = 16
GRID_X = PAD
GRID_Y = HEADER_H + PAD
GRID_W = 690
GRID_H = WIN_H - GRID_Y - PAD
PANEL_X = GRID_X + GRID_W + PAD
PANEL_W = WIN_W - PANEL_X - PAD

# ------------------------- Einstellbare Presets ---------------------------- #
POP_OPTIONS = [30, 50, 100, 150, 200, 300]
# (Name, rate, strength)
MUTATION_PRESETS = [
    ("Niedrig", 0.03, 0.12),
    ("Mittel", 0.05, 0.20),
    ("Hoch", 0.10, 0.35),
]
FRUIT_OPTIONS = [1, 2, 3, 5]
VISIBLE_OPTIONS = [12, 20, 30, 50]
SPEED_LEVELS = [10, 20, 40, 80, 160]  # Zuege pro Sekunde in der Live-Ansicht

# ----------------------------- Extra-Farben -------------------------------- #
MINI_BG = (22, 25, 35)
SNAKE_HEAD = Palette.SNAKE_HEAD
SNAKE_BODY = Palette.SNAKE_BODY
FRUIT = Palette.FRUIT
ALIVE_BORDER = (48, 60, 58)
DEAD_BORDER = (60, 40, 44)
ELITE_BORDER = Palette.ACCENT
CURVE_MEAN = Palette.ACCENT
CURVE_BEST = (250, 200, 90)


class Dashboard:
    MENU = "MENU"
    RUNNING = "RUNNING"

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Snake — Neuroevolution Dashboard")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

        self.f_title = load_font(30, "semibold")
        self.f_h2 = load_font(20, "semibold")
        self.f_label = load_font(13, "semibold")
        self.f_value = load_font(22, "semibold")
        self.f_body = load_font(16, "regular")
        self.f_small = load_font(14, "regular")
        self.f_tiny = load_font(11, "regular")

        # Menue-Auswahl (Indizes in die Preset-Listen).
        self.pop_idx = 2       # 100
        self.mut_idx = 1       # Mittel
        self.fruit_idx = 0     # 1
        self.visible_idx = 1   # 20
        self.menu_row = 0

        self.state = Dashboard.MENU
        self.running = True

        # Trainingszustand (wird beim Start gesetzt).
        self.trainer: EvolutionTrainer | None = None
        self.cfg: EvolutionConfig | None = None
        self.visible_count = 20
        self.speed_idx = 2     # 40 Zuege/s
        self.turbo = False
        self.paused = False
        self.move_accum = 0.0
        self.last_stats: GenerationStats | None = None
        self.gen_times: list[float] = []
        self._last_gen_time = time.time()

    # ================================================================== #
    # Hauptschleife
    # ================================================================== #
    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60)
            self._handle_events()
            if self.state == Dashboard.RUNNING:
                self._update(dt)
                self._draw_running()
            else:
                self._draw_menu()
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
                if self.state == Dashboard.MENU:
                    self._on_menu_key(event.key)
                else:
                    self._on_run_key(event.key)

    def _on_menu_key(self, key: int) -> None:
        if key in (pygame.K_UP, pygame.K_w):
            self.menu_row = (self.menu_row - 1) % 5
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu_row = (self.menu_row + 1) % 5
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._menu_adjust(-1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._menu_adjust(+1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self._start_training()
        elif key == pygame.K_ESCAPE:
            self.running = False

    def _menu_adjust(self, delta: int) -> None:
        if self.menu_row == 0:
            self.pop_idx = (self.pop_idx + delta) % len(POP_OPTIONS)
        elif self.menu_row == 1:
            self.mut_idx = (self.mut_idx + delta) % len(MUTATION_PRESETS)
        elif self.menu_row == 2:
            self.fruit_idx = (self.fruit_idx + delta) % len(FRUIT_OPTIONS)
        elif self.menu_row == 3:
            self.visible_idx = (self.visible_idx + delta) % len(VISIBLE_OPTIONS)

    def _on_run_key(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_t:
            self.turbo = not self.turbo
        elif key in (pygame.K_UP, pygame.K_KP_PLUS, pygame.K_PLUS):
            self.speed_idx = min(self.speed_idx + 1, len(SPEED_LEVELS) - 1)
        elif key in (pygame.K_DOWN, pygame.K_KP_MINUS, pygame.K_MINUS):
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif key == pygame.K_ESCAPE:
            self.state = Dashboard.MENU

    # ================================================================== #
    # Training starten
    # ================================================================== #
    def _start_training(self) -> None:
        _, rate, strength = MUTATION_PRESETS[self.mut_idx]
        self.cfg = EvolutionConfig(
            population_size=POP_OPTIONS[self.pop_idx],
            hidden=DEFAULT_HIDDEN,
            mutation_rate=rate,
            mutation_strength=strength,
            fruit_count=FRUIT_OPTIONS[self.fruit_idx],
        )
        self.visible_count = min(VISIBLE_OPTIONS[self.visible_idx], self.cfg.population_size)
        self.trainer = EvolutionTrainer(self.cfg, log_to_csv=True)
        self.trainer.begin_generation()
        self.turbo = False
        self.paused = False
        self.move_accum = 0.0
        self.last_stats = None
        self.gen_times = []
        self._last_gen_time = time.time()
        self.state = Dashboard.RUNNING

    def _on_generation_end(self) -> None:
        self.last_stats = self.trainer.end_generation()
        now = time.time()
        self.gen_times.append(now - self._last_gen_time)
        self._last_gen_time = now
        self.gen_times = self.gen_times[-20:]

    # ================================================================== #
    # Training vorantreiben
    # ================================================================== #
    def _update(self, dt: int) -> None:
        if self.paused or self.trainer is None:
            return

        if self.turbo:
            # So viele Generationen wie in ein kleines Zeitbudget passen ->
            # maximale Geschwindigkeit, UI bleibt trotzdem reaktiv.
            budget = 0.05
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < budget:
                if self.trainer.generation_active:
                    while not self.trainer.step_generation():
                        pass
                    self._on_generation_end()
                else:
                    self.trainer.begin_generation()
        else:
            if not self.trainer.generation_active:
                self.trainer.begin_generation()
            self.move_accum += dt
            interval = 1000.0 / SPEED_LEVELS[self.speed_idx]
            steps = 0
            while self.move_accum >= interval and steps < 200:
                self.move_accum -= interval
                steps += 1
                if self.trainer.step_generation():
                    self._on_generation_end()
                    self.move_accum = 0.0
                    break

    # ================================================================== #
    # Zeichnen: Menue
    # ================================================================== #
    def _draw_menu(self) -> None:
        self.screen.fill(Palette.BG)
        cx = WIN_W // 2
        self._center(self.f_title, "Neuroevolution — Training", Palette.ACCENT, cx, 90)
        self._center(self.f_small,
                     "100 Schlangen lernen Snake von selbst — durch Zucht über Generationen",
                     Palette.TEXT_DIM, cx, 128)

        _, rate, strength = MUTATION_PRESETS[self.mut_idx]
        entries = [
            ("Populationsgröße", str(POP_OPTIONS[self.pop_idx])),
            ("Mutation", f"{MUTATION_PRESETS[self.mut_idx][0]}  (Rate {rate}, Stärke {strength})"),
            ("Früchte", str(FRUIT_OPTIONS[self.fruit_idx])),
            ("Sichtbare Schlangen", str(VISIBLE_OPTIONS[self.visible_idx])),
            ("TRAINING STARTEN", None),
        ]
        px, pw, row_h, gap, y0 = 300, 580, 52, 12, 200
        for i, (label, value) in enumerate(entries):
            y = y0 + i * (row_h + gap)
            selected = (i == self.menu_row)
            rect = pygame.Rect(px, y, pw, row_h)
            if value is None:
                bg = Palette.ACCENT if selected else (30, 35, 50)
                pygame.draw.rect(self.screen, bg, rect, border_radius=12)
                if not selected:
                    pygame.draw.rect(self.screen, Palette.BORDER, rect, width=2, border_radius=12)
                self._center(self.f_h2, label, Palette.BG if selected else Palette.TEXT,
                             rect.centerx, rect.centery)
            else:
                if selected:
                    pygame.draw.rect(self.screen, (26, 38, 34), rect, border_radius=12)
                    pygame.draw.rect(self.screen, Palette.ACCENT, rect, width=2, border_radius=12)
                lc = Palette.TEXT if selected else Palette.TEXT_DIM
                vc = Palette.ACCENT if selected else Palette.TEXT
                ls = self.f_body.render(label, lc)
                self.screen.blit(ls, ls.get_rect(midleft=(px + 22, rect.centery)))
                vs = self.f_body.render(value, vc)
                self.screen.blit(vs, vs.get_rect(midright=(px + pw - 22, rect.centery)))

        self._center(self.f_tiny,
                     "Pfeil hoch/runter wählen   ·   links/rechts ändern   ·   Enter startet",
                     Palette.TEXT_DIM, cx, WIN_H - 40)

    # ================================================================== #
    # Zeichnen: laufendes Training
    # ================================================================== #
    def _draw_running(self) -> None:
        self.screen.fill(Palette.BG)
        self._draw_header()
        if self.turbo:
            self._draw_turbo_placeholder()
        else:
            self._draw_grid()
        self._draw_panel()

    def _draw_header(self) -> None:
        pygame.draw.rect(self.screen, Palette.HEADER_BG, pygame.Rect(0, 0, WIN_W, HEADER_H))
        pygame.draw.line(self.screen, Palette.BORDER, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), 2)

        gen = self.trainer.population.generation if self.trainer else 0
        title = self.f_h2.render(f"Generation {gen}", Palette.TEXT)
        self.screen.blit(title, (PAD, 18))

        # Statusanzeige rechts: Tempo / Turbo / Pause.
        if self.turbo:
            status, col = "TURBO", CURVE_BEST
        elif self.paused:
            status, col = "PAUSE", Palette.ACCENT_WARN
        else:
            status, col = f"Tempo {SPEED_LEVELS[self.speed_idx]}/s", Palette.TEXT_DIM
        ss = self.f_body.render(status, col)
        self.screen.blit(ss, ss.get_rect(midright=(WIN_W - PAD, HEADER_H // 2)))

        hint = self.f_tiny.render(
            "Leertaste Pause   ·   T Turbo   ·   hoch/runter Tempo   ·   Esc Menü",
            Palette.TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, HEADER_H // 2)))

    def _draw_turbo_placeholder(self) -> None:
        rect = pygame.Rect(GRID_X, GRID_Y, GRID_W, GRID_H)
        pygame.draw.rect(self.screen, MINI_BG, rect, border_radius=8)
        self._center(self.f_title, "TURBO", CURVE_BEST, rect.centerx, rect.centery - 20)
        self._center(self.f_small, "Anzeige aus — es wird mit voller Geschwindigkeit gerechnet.",
                     Palette.TEXT_DIM, rect.centerx, rect.centery + 20)
        self._center(self.f_small, "T drücken, um wieder zuzuschauen.",
                     Palette.TEXT_DIM, rect.centerx, rect.centery + 44)

    def _draw_grid(self) -> None:
        if not self.trainer or not self.trainer.games:
            return
        n = self.visible_count
        gc = math.ceil(math.sqrt(n * GRID_W / GRID_H))
        gr = math.ceil(n / gc)
        cell_w = (GRID_W - (gc - 1) * 6) / gc
        cell_h = (GRID_H - (gr - 1) * 6) / gr

        elite = self.trainer.cfg.elitism
        for idx in range(min(n, len(self.trainer.games))):
            r, c = divmod(idx, gc)
            x = GRID_X + c * (cell_w + 6)
            y = GRID_Y + r * (cell_h + 6)
            self._draw_mini(self.trainer.games[idx],
                            pygame.Rect(int(x), int(y), int(cell_w), int(cell_h)),
                            is_elite=(idx < elite))

    def _draw_mini(self, game, rect: pygame.Rect, is_elite: bool) -> None:
        pygame.draw.rect(self.screen, MINI_BG, rect, border_radius=4)
        cw = rect.width / game.cols
        ch = rect.height / game.rows
        csize = max(1, int(math.ceil(min(cw, ch))))

        for (fx, fy) in game.fruits:
            px = rect.x + fx * cw
            py = rect.y + fy * ch
            pygame.draw.rect(self.screen, FRUIT, (int(px), int(py), csize, csize))

        for i, (sx, sy) in enumerate(game.snake):
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            px = rect.x + sx * cw
            py = rect.y + sy * ch
            pygame.draw.rect(self.screen, color, (int(px), int(py), csize, csize))

        if is_elite:
            border = ELITE_BORDER
        elif game.alive:
            border = ALIVE_BORDER
        else:
            border = DEAD_BORDER
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=4)

        if rect.width > 70:
            sc = self.f_tiny.render(str(game.score), Palette.TEXT)
            self.screen.blit(sc, (rect.x + 4, rect.y + 3))

    # ================================================================== #
    # Zeichnen: Statistik-Panel + Lernkurve
    # ================================================================== #
    def _draw_panel(self) -> None:
        s = self.last_stats
        panel = pygame.Rect(PANEL_X, GRID_Y, PANEL_W, GRID_H)
        pygame.draw.rect(self.screen, Palette.HEADER_BG, panel, border_radius=8)

        x = PANEL_X + 18
        y = GRID_Y + 16
        self._text(self.f_label, "STATISTIK", Palette.TEXT_DIM, x, y)
        y += 26

        if s is None:
            self._text(self.f_small, "Erste Generation laeuft ...", Palette.TEXT_DIM, x, y)
        else:
            # Kennzahlenblock (zweispaltig).
            col2 = PANEL_X + PANEL_W // 2 + 4
            self._stat_pair(x, col2, y, "Bester Score", str(s.best_score),
                            "Bestwert gesamt", str(s.alltime_best_score),
                            Palette.ACCENT, CURVE_BEST)
            y += 54
            self._stat_pair(x, col2, y, "Ø Score", f"{s.mean_score:.2f}",
                            "Ø Länge", f"{s.mean_length:.1f}",
                            Palette.TEXT, Palette.TEXT)
            y += 54
            self._stat_pair(x, col2, y, "Ø Schritte", f"{s.mean_steps:.0f}",
                            "Diversität", f"{s.diversity:.3f}",
                            Palette.TEXT, Palette.TEXT)
            y += 60

            # Todesursachen als Balken.
            self._text(self.f_label, "TODESURSACHEN (diese Gen)", Palette.TEXT_DIM, x, y)
            y += 22
            total = max(1, sum(s.deaths.values()))
            causes = [
                ("Wand", s.deaths.get("wall", 0), (231, 111, 81)),
                ("Selbst", s.deaths.get("self", 0), (233, 196, 106)),
                ("Verhungert", s.deaths.get("starvation", 0), (109, 158, 235)),
                ("Überlebt", s.deaths.get("timeout", 0) + s.deaths.get("won", 0), Palette.ACCENT),
            ]
            for name, count, color in causes:
                self._death_bar(x, y, PANEL_W - 36, name, count, count / total, color)
                y += 26
            y += 6

            # Tempo-Info.
            if self.gen_times:
                gps = 1.0 / (sum(self.gen_times) / len(self.gen_times))
                self._text(self.f_small, f"Tempo: {gps:.1f} Generationen/s",
                           Palette.TEXT_DIM, x, y)
            y += 26

        # Lernkurve unten im Panel.
        curve = pygame.Rect(PANEL_X + 14, GRID_Y + GRID_H - 190, PANEL_W - 28, 172)
        self._draw_curve(curve)

    def _stat_pair(self, x1, x2, y, l1, v1, l2, v2, c1, c2) -> None:
        self._text(self.f_label, l1, Palette.TEXT_DIM, x1, y)
        self._text(self.f_value, v1, c1, x1, y + 16)
        self._text(self.f_label, l2, Palette.TEXT_DIM, x2, y)
        self._text(self.f_value, v2, c2, x2, y + 16)

    def _death_bar(self, x, y, w, name, count, frac, color) -> None:
        label = self.f_small.render(f"{name}", Palette.TEXT)
        self.screen.blit(label, (x, y))
        bar_x = x + 96
        bar_w = w - 96 - 34
        pygame.draw.rect(self.screen, (32, 36, 48), (bar_x, y + 2, bar_w, 12), border_radius=6)
        if frac > 0:
            pygame.draw.rect(self.screen, color,
                             (bar_x, y + 2, max(2, int(bar_w * frac)), 12), border_radius=6)
        cnt = self.f_small.render(str(count), Palette.TEXT_DIM)
        self.screen.blit(cnt, cnt.get_rect(midright=(x + w, y + 8)))

    def _draw_curve(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, MINI_BG, rect, border_radius=6)
        self._text(self.f_label, "LERNKURVE (Score / Generation)", Palette.TEXT_DIM,
                   rect.x + 8, rect.y + 6)
        plot = pygame.Rect(rect.x + 34, rect.y + 28, rect.width - 44, rect.height - 46)

        hist = self.trainer.history if self.trainer else []
        if len(hist) < 2:
            self._text(self.f_tiny, "sammelt Daten ...", Palette.TEXT_DIM, plot.x, plot.y + 6)
            return

        mean_vals = [h.mean_score for h in hist]
        best_vals = [h.alltime_best_score for h in hist]
        vmax = max(1.0, max(best_vals))
        n = len(hist)

        def pt(i, val):
            px = plot.x + (i / (n - 1)) * plot.width
            py = plot.y + plot.height - (val / vmax) * plot.height
            return (px, py)

        # Achsenlinien.
        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.y), (plot.x, plot.y + plot.height), 1)
        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.y + plot.height), (plot.x + plot.width, plot.y + plot.height), 1)
        # y-Skala Beschriftung (max).
        self._text(self.f_tiny, str(int(vmax)), Palette.TEXT_DIM, rect.x + 6, plot.y - 4)

        best_line = [pt(i, v) for i, v in enumerate(best_vals)]
        mean_line = [pt(i, v) for i, v in enumerate(mean_vals)]
        if len(best_line) >= 2:
            pygame.draw.lines(self.screen, CURVE_BEST, False, best_line, 2)
        if len(mean_line) >= 2:
            pygame.draw.lines(self.screen, CURVE_MEAN, False, mean_line, 2)

        # kleine Legende.
        self._text(self.f_tiny, "Bestwert", CURVE_BEST, plot.x + 6, plot.y + 2)
        self._text(self.f_tiny, "Ø Score", CURVE_MEAN, plot.x + 70, plot.y + 2)

    # ------------------------------------------------------------------ #
    # kleine Text-Helfer
    # ------------------------------------------------------------------ #
    def _text(self, font, text, color, x, y) -> None:
        self.screen.blit(font.render(text, color), (x, y))

    def _center(self, font, text, color, cx, cy) -> None:
        surf = font.render(text, color)
        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))


def main() -> None:
    Dashboard().run()


if __name__ == "__main__":
    main()
