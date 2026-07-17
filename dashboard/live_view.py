"""Live-Dashboard fuer das Neuroevolution-Training -- reine Auswertungsansicht.

Zeigt in Echtzeit:
- ein STATISTIK-Panel (Generation, Score, Fitness, Effizienz, Stagnation, Diversitaet)
- die Todesursachen als Balken
- eine grosse LERNKURVE (Score-Entwicklung ueber die Generationen)

WICHTIG (bewusste Design-Entscheidung): Es gibt hier KEINE Animation einzelner
Schlangen mehr. Das Training laeuft immer mit maximaler Rechengeschwindigkeit --
ohne an eine Bildwiederholrate oder Zug-Geschwindigkeit gekoppelt zu sein. Wer
zusehen will, wie ein trainiertes Netz WIRKLICH spielt (mit huebscher Grafik),
nutzt watch_ai.py -- das laedt den gespeicherten Champion und zeigt ihn im
normalen, schoenen Spielfenster. Hier im Trainingsfenster zaehlt nur: so viele
Generationen wie moeglich pro Sekunde, mit klarer Auswertung, was gerade passiert.

Steuerung waehrend des Trainings:
- Leertaste : Pause / weiter
- Esc       : zurueck ins Menue

Das Dashboard "treibt" den Trainer (ai/evolution/train_evolution.py) in grossen
Zeitbloecken an -- die Rechenlogik selbst enthaelt kein pygame.
"""

from __future__ import annotations

import time

import pygame

from game.config import Palette
from game.fonts import load_font
from ai.network import DEFAULT_HIDDEN
from ai.evolution.train_evolution import EvolutionConfig, EvolutionTrainer, GenerationStats

# ----------------------------- Fenster-Layout ------------------------------ #
WIN_W, WIN_H = 1100, 760
HEADER_H = 60
PAD = 20

# ------------------------- Einstellbare Presets ---------------------------- #
POP_OPTIONS = [30, 50, 100, 150, 200, 300]
# (Name, rate, strength)
MUTATION_PRESETS = [
    ("Niedrig", 0.03, 0.12),
    ("Mittel", 0.05, 0.20),
    ("Hoch", 0.10, 0.35),
]
FRUIT_OPTIONS = list(range(1, 11))  # 1 bis 10, wie im Menschen-Spiel

# Bewertungs-Tiefe: wie viele unabhaengige Partien (mit je eigenem Zufalls-
# Fruchtlayout) jedes Genom pro Generation spielt, bevor seine Fitness aus dem
# DURCHSCHNITT gebildet wird. Mehr Partien = robustere, weniger vom Zufall
# abhaengige Bewertung (verhindert, dass ein Genom nur durch ein gluecklich
# leichtes Layout hoch bewertet wird), kostet aber proportional mehr Rechenzeit.
EPISODES_PRESETS = [
    ("Schnell (1x)", 1),
    ("Ausgewogen (3x)", 3),
    ("Robust (5x)", 5),
]
# Elite-Anteil: so viele der besten Genome kommen JEDE Generation unveraendert
# in die naechste Runde. Schuetzt davor, dass ein bereits gefundenes gutes
# Genom durch Mutation/Zufall wieder verloren geht.
ELITISM_PRESETS = [
    ("Wenig (2%)", 0.02),
    ("Mittel (5%)", 0.05),
    ("Viel (10%)", 0.10),
]

# ----------------------------- Extra-Farben -------------------------------- #
PANEL_BG = (22, 25, 35)
CURVE_MEAN = Palette.ACCENT
CURVE_BEST = (250, 200, 90)
STAGNATION_OK = Palette.ACCENT
STAGNATION_WARN = (233, 196, 106)
STAGNATION_BAD = Palette.ACCENT_WARN


class Dashboard:
    MENU = "MENU"
    RUNNING = "RUNNING"

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Snake — Neuroevolution Training")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

        self.f_title = load_font(30, "semibold")
        self.f_h2 = load_font(20, "semibold")
        self.f_label = load_font(13, "semibold")
        self.f_value = load_font(30, "semibold")
        self.f_body = load_font(16, "regular")
        self.f_small = load_font(14, "regular")
        self.f_tiny = load_font(11, "regular")

        # Menue-Auswahl (Indizes in die Preset-Listen).
        self.pop_idx = 2       # 100
        self.mut_idx = 1       # Mittel
        self.fruit_idx = 0     # 1
        self.episodes_idx = 1  # Ausgewogen (3x)
        self.elitism_idx = 1   # Mittel (5%)
        self.menu_row = 0
        self._menu_row_count = 6  # 5 Einstellungen + START

        self.state = Dashboard.MENU
        self.running = True

        # Trainingszustand (wird beim Start gesetzt).
        self.trainer: EvolutionTrainer | None = None
        self.cfg: EvolutionConfig | None = None
        self.paused = False
        self.last_stats: GenerationStats | None = None
        self.gen_times: list[float] = []
        self._last_gen_time = time.time()

    # ================================================================== #
    # Hauptschleife
    # ================================================================== #
    def run(self) -> None:
        while self.running:
            self.clock.tick(60)
            self._handle_events()
            if self.state == Dashboard.RUNNING:
                self._update()
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
            self.menu_row = (self.menu_row - 1) % self._menu_row_count
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu_row = (self.menu_row + 1) % self._menu_row_count
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
            self.episodes_idx = (self.episodes_idx + delta) % len(EPISODES_PRESETS)
        elif self.menu_row == 4:
            self.elitism_idx = (self.elitism_idx + delta) % len(ELITISM_PRESETS)

    def _on_run_key(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_ESCAPE:
            self.state = Dashboard.MENU

    # ================================================================== #
    # Training starten
    # ================================================================== #
    def _start_training(self) -> None:
        _, rate, strength = MUTATION_PRESETS[self.mut_idx]
        pop_size = POP_OPTIONS[self.pop_idx]
        _, elite_frac = ELITISM_PRESETS[self.elitism_idx]
        elite_count = max(1, round(pop_size * elite_frac))
        self.cfg = EvolutionConfig(
            population_size=pop_size,
            hidden=DEFAULT_HIDDEN,
            elitism=elite_count,
            mutation_rate=rate,
            mutation_strength=strength,
            fruit_count=FRUIT_OPTIONS[self.fruit_idx],
            episodes_per_genome=EPISODES_PRESETS[self.episodes_idx][1],
        )
        self.trainer = EvolutionTrainer(self.cfg, log_to_csv=True)
        self.trainer.begin_generation()
        self.paused = False
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
    # Training vorantreiben -- IMMER mit maximaler Rechengeschwindigkeit
    # ================================================================== #
    def _update(self) -> None:
        """Rechnet so viele Generationen wie in ein Zeitbudget passen.

        Keine Kopplung an eine Zug-Geschwindigkeit oder Bildwiederholrate mehr
        (es gibt ja nichts mehr zu animieren) -- nur ein kleines Zeitbudget pro
        Frame, damit das Fenster weiterhin auf Leertaste/Esc reagiert, waehrend
        im Hintergrund mit voller Geschwindigkeit gerechnet wird.
        """
        if self.paused or self.trainer is None:
            return

        budget = 0.1
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < budget:
            if self.trainer.generation_active:
                while not self.trainer.step_generation():
                    pass
                self._on_generation_end()
            else:
                self.trainer.begin_generation()

    # ================================================================== #
    # Zeichnen: Menue
    # ================================================================== #
    def _draw_menu(self) -> None:
        self.screen.fill(Palette.BG)
        cx = WIN_W // 2
        self._center(self.f_title, "Neuroevolution — Training", Palette.ACCENT, cx, 80)
        self._center(self.f_small,
                     "Schlangen lernen Snake von selbst — durch Zucht über Generationen",
                     Palette.TEXT_DIM, cx, 116)

        _, rate, strength = MUTATION_PRESETS[self.mut_idx]
        entries = [
            ("Populationsgröße", str(POP_OPTIONS[self.pop_idx])),
            ("Mutation", f"{MUTATION_PRESETS[self.mut_idx][0]}  (Rate {rate}, Stärke {strength})"),
            ("Früchte", str(FRUIT_OPTIONS[self.fruit_idx])),
            ("Bewertungstiefe", EPISODES_PRESETS[self.episodes_idx][0]),
            ("Elite-Anteil", ELITISM_PRESETS[self.elitism_idx][0]),
            ("TRAINING STARTEN", None),
        ]
        px, pw, row_h, gap, y0 = 260, 580, 50, 11, 168
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
                     Palette.TEXT_DIM, cx, WIN_H - 30)

    # ================================================================== #
    # Zeichnen: laufendes Training (reine Auswertung, kein Grid)
    # ================================================================== #
    def _draw_running(self) -> None:
        self.screen.fill(Palette.BG)
        self._draw_header()

        content = pygame.Rect(PAD, HEADER_H + PAD, WIN_W - 2 * PAD, WIN_H - HEADER_H - 2 * PAD)
        s = self.last_stats
        if s is None:
            self._center(self.f_body, "Erste Generation läuft ...", Palette.TEXT_DIM,
                        content.centerx, content.y + 40)
            return

        self._draw_stat_tiles(content, s)
        self._draw_death_bars(content, s)
        self._draw_curve(content)

    def _draw_header(self) -> None:
        pygame.draw.rect(self.screen, Palette.HEADER_BG, pygame.Rect(0, 0, WIN_W, HEADER_H))
        pygame.draw.line(self.screen, Palette.BORDER, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), 2)

        gen = self.trainer.population.generation if self.trainer else 0
        title = self.f_h2.render(f"Generation {gen}", Palette.TEXT)
        self.screen.blit(title, (PAD, 18))

        status = "PAUSE" if self.paused else "LÄUFT"
        col = Palette.ACCENT_WARN if self.paused else Palette.ACCENT
        gps_txt = ""
        if self.gen_times:
            gps = 1.0 / (sum(self.gen_times) / len(self.gen_times))
            gps_txt = f"   ·   {gps:.2f} Gen/s"
        ss = self.f_body.render(status + gps_txt, col)
        self.screen.blit(ss, ss.get_rect(midright=(WIN_W - PAD, HEADER_H // 2)))

        hint = self.f_tiny.render("Leertaste Pause   ·   Esc Menü", Palette.TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, HEADER_H // 2)))

    def _draw_stat_tiles(self, content: pygame.Rect, s: GenerationStats) -> None:
        """Acht grosse Kennzahlen-Kacheln (4 Spalten x 2 Reihen)."""
        eff_txt = f"{s.mean_steps_per_fruit:.1f}" if s.mean_steps_per_fruit is not None else "—"
        stag = s.generations_since_improvement
        stag_color = STAGNATION_OK if stag < 15 else (STAGNATION_WARN if stag < 40 else STAGNATION_BAD)

        tiles = [
            ("CHAMPION Ø", f"{s.best_avg_score:.2f}", Palette.ACCENT),
            ("BESTWERT (EINZELPARTIE)", str(s.alltime_best_score), CURVE_BEST),
            ("Ø SCORE", f"{s.mean_score:.2f}", Palette.TEXT),
            ("Ø LÄNGE", f"{s.mean_length:.1f}", Palette.TEXT),
            ("Ø SCHRITTE", f"{s.mean_steps:.0f}", Palette.TEXT),
            ("EFFIZIENZ (SCHR./FRUCHT)", eff_txt, Palette.TEXT),
            ("DIVERSITÄT", f"{s.diversity:.3f}", Palette.TEXT),
            ("STAGNATION", f"{stag} Gen", stag_color),
        ]
        cols, rows = 4, 2
        gap = 14
        tile_w = (content.width - (cols - 1) * gap) / cols
        tile_h = 92
        for i, (label, value, color) in enumerate(tiles):
            r, c = divmod(i, cols)
            x = content.x + c * (tile_w + gap)
            y = content.y + r * (tile_h + gap)
            rect = pygame.Rect(int(x), int(y), int(tile_w), tile_h)
            pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
            self._center(self.f_label, label, Palette.TEXT_DIM, rect.centerx, rect.y + 22)
            self._center(self.f_value, value, color, rect.centerx, rect.y + 58)

        self._tiles_bottom = content.y + rows * tile_h + (rows - 1) * gap

    def _draw_death_bars(self, content: pygame.Rect, s: GenerationStats) -> None:
        y = self._tiles_bottom + 22
        self._text(self.f_label, "TODESURSACHEN (diese Generation)", Palette.TEXT_DIM, content.x, y)
        y += 24
        total = max(1, sum(s.deaths.values()))
        causes = [
            ("Wand", s.deaths.get("wall", 0), (231, 111, 81)),
            ("Selbst", s.deaths.get("self", 0), (233, 196, 106)),
            ("Verhungert", s.deaths.get("starvation", 0), (109, 158, 235)),
            ("Überlebt", s.deaths.get("timeout", 0) + s.deaths.get("won", 0), Palette.ACCENT),
        ]
        col_w = content.width / 2
        for i, (name, count, color) in enumerate(causes):
            r, c = divmod(i, 2)
            x = content.x + c * col_w
            yy = y + r * 26
            self._death_bar(int(x), yy, int(col_w - 24), name, count, count / total, color)

        self._death_bottom = y + 2 * 26 + 10

    def _death_bar(self, x, y, w, name, count, frac, color) -> None:
        label = self.f_small.render(f"{name}", Palette.TEXT)
        self.screen.blit(label, (x, y))
        bar_x = x + 130
        bar_w = w - 130 - 40
        pygame.draw.rect(self.screen, (32, 36, 48), (bar_x, y + 2, bar_w, 12), border_radius=6)
        if frac > 0:
            pygame.draw.rect(self.screen, color,
                             (bar_x, y + 2, max(2, int(bar_w * frac)), 12), border_radius=6)
        cnt = self.f_small.render(str(count), Palette.TEXT_DIM)
        self.screen.blit(cnt, cnt.get_rect(midright=(x + w, y + 8)))

    def _draw_curve(self, content: pygame.Rect) -> None:
        rect = pygame.Rect(content.x, self._death_bottom, content.width,
                           content.bottom - self._death_bottom)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
        self._text(self.f_label, "LERNKURVE (Score / Generation)", Palette.TEXT_DIM,
                   rect.x + 12, rect.y + 10)
        plot = pygame.Rect(rect.x + 40, rect.y + 36, rect.width - 56, rect.height - 56)

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

        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.y), (plot.x, plot.y + plot.height), 1)
        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.y + plot.height), (plot.x + plot.width, plot.y + plot.height), 1)
        self._text(self.f_tiny, str(int(vmax)), Palette.TEXT_DIM, rect.x + 8, plot.y - 4)

        best_line = [pt(i, v) for i, v in enumerate(best_vals)]
        mean_line = [pt(i, v) for i, v in enumerate(mean_vals)]
        if len(best_line) >= 2:
            pygame.draw.lines(self.screen, CURVE_BEST, False, best_line, 2)
        if len(mean_line) >= 2:
            pygame.draw.lines(self.screen, CURVE_MEAN, False, mean_line, 2)

        self._text(self.f_tiny, "Bestwert", CURVE_BEST, plot.x + 6, plot.y + 4)
        self._text(self.f_tiny, "Ø Score", CURVE_MEAN, plot.x + 70, plot.y + 4)

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
