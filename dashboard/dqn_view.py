"""Live-Fenster fuer das DQN-Training: mehrere Spiele + Statistik + Lernkurve.

Anders als beim Neuroevolution-Dashboard (dashboard/live_view.py) siehst du hier
die Schlangen WIRKLICH spielen, waehrend sie lernen. Das ist bei DQN sinnvoll,
weil hier ein einziges Gehirn dauerhaft weitertrainiert -- man sieht also live,
wie aus zappeligem Unsinn allmaehlich zielgerichtetes Spiel wird.

Aufbau des Fensters:
- oben: die parallel laufenden Spielfelder (klein, aber echt -- dieselbe Engine)
- Mitte: acht Kennzahlen-Kacheln
- darunter: Todesursachen als Balken
- unten: die LERNKURVE (Ø-Score ueber die Zeit) -- selbst gezeichnet mit
  pygame.draw.lines, bewusst ohne matplotlib.

Steuerung:
    Leertaste  Pause / weiter
    Pfeil links/rechts oder +/-   Geschwindigkeit (Zuege pro Bild)
    T          TURBO an/aus  (rechnet mit voller Kraft, zeichnet keine Felder)
    Esc        zurueck ins Menue (das Training bleibt erhalten)

Warum Geschwindigkeit und Zeichnen getrennt sind:
Das Zeichnen kostet Zeit, die nicht ins Lernen fliesst. Deshalb entscheidet eine
einzige Zahl -- "wie viele Trainings-Ticks pro gezeichnetem Bild" -- wie schnell
gelernt wird. Im Turbo-Modus rechnen wir stattdessen ein festes Zeitbudget lang,
so viel wie in dieses Budget passt, und zeichnen nur noch die Zahlen. Das ist mit
Abstand der schnellste Lernmodus.

Wichtig fuer Python 3.14: Schriften kommen NUR ueber game.fonts.load_font --
pygame.font stuerzt dort beim Import ab (bekannter pygame-Bug).
"""

from __future__ import annotations

import math
import time
from collections import deque

import pygame

from game.config import Palette
from game.fonts import load_font
from ai.dqn.config import DQNConfig
from ai.dqn.trainer import DQNStats, MultiGameTrainer

# ----------------------------- Fenster-Layout ------------------------------ #
WIN_W, WIN_H = 1320, 860
HEADER_H = 60
PAD = 20
BOARD_AREA_H = 246       # Hoehe des Bereichs mit den Spielfeldern
BOARD_GAP = 14
CAPTION_H = 18           # kleine Zeile ueber jedem Feld (Score/Laenge)

# ------------------------- Einstellbare Presets ---------------------------- #
# Alles hier ist nur die MENUE-Auswahl. Die eigentliche Wahrheit steht in
# ai/dqn/config.py -- dort kann man jeden Wert noch feiner drehen.
NUM_GAMES_OPTIONS = [1, 2, 3, 4, 5, 6, 8]
HIDDEN_PRESETS = [
    ("Klein (64, 64)", (64, 64)),
    ("Mittel (128, 128)", (128, 128)),
    ("Groß (256, 128)", (256, 128)),
    ("Sehr groß (256, 256)", (256, 256)),
]
LR_PRESETS = [
    ("Vorsichtig (0.0003)", 3e-4),
    ("Normal (0.001)", 1e-3),
    ("Forsch (0.002)", 2e-3),
]
EPS_PRESETS = [
    ("Kurz (20k Ticks)", 20_000),
    ("Normal (40k Ticks)", 40_000),
    ("Gründlich (80k Ticks)", 80_000),
    ("Sehr gründlich (150k)", 150_000),
]
GAMMA_PRESETS = [
    ("Kurzsichtig (0.85)", 0.85),
    ("Normal (0.90)", 0.90),
    ("Weitsichtig (0.95)", 0.95),
    ("Sehr weitsichtig (0.99)", 0.99),
]
FRUIT_OPTIONS = list(range(1, 11))

# Wie viele Trainings-Ticks pro gezeichnetem Bild gerechnet werden.
SPEED_LEVELS = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_SPEED = 2        # -> 4 Ticks pro Bild
TURBO_BUDGET = 0.06      # Sekunden Rechenzeit pro Bild im Turbo-Modus

# ----------------------------- Extra-Farben -------------------------------- #
PANEL_BG = (22, 25, 35)
CURVE_MEAN = Palette.ACCENT
CURVE_BEST = (250, 200, 90)
TURBO_COLOR = (250, 200, 90)


class DQNDashboard:
    MENU = "MENU"
    RUNNING = "RUNNING"

    def __init__(self, cfg: DQNConfig | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Snake — Deep Q-Learning Training")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

        self.f_title = load_font(30, "semibold")
        self.f_h2 = load_font(20, "semibold")
        self.f_label = load_font(13, "semibold")
        self.f_value = load_font(28, "semibold")
        self.f_body = load_font(16, "regular")
        self.f_small = load_font(14, "regular")
        self.f_tiny = load_font(11, "regular")

        # Basis-Konfiguration; das Menue ueberschreibt daraus nur einzelne Werte.
        self.base_cfg = cfg or DQNConfig()

        # Menue-Auswahl (Indizes in die Preset-Listen).
        self.games_idx = NUM_GAMES_OPTIONS.index(self.base_cfg.num_games) \
            if self.base_cfg.num_games in NUM_GAMES_OPTIONS else 4
        self.hidden_idx = 1
        self.lr_idx = 1
        self.eps_idx = 1
        self.gamma_idx = 1
        self.fruit_idx = self.base_cfg.fruit_count - 1
        self.menu_row = 0

        self.state = DQNDashboard.MENU
        self.running = True

        # Trainingszustand.
        self.trainer: MultiGameTrainer | None = None
        self.cfg: DQNConfig | None = None
        self.paused = False
        self.turbo = False
        self.speed_idx = DEFAULT_SPEED

        # Durchsatz-Messung (Zuege pro Sekunde) ueber ein gleitendes Fenster.
        self._perf: deque[tuple[float, int]] = deque(maxlen=40)

    # ================================================================== #
    # Hauptschleife
    # ================================================================== #
    def run(self) -> None:
        while self.running:
            self.clock.tick(60)
            self._handle_events()
            if self.state == DQNDashboard.RUNNING:
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
                if self.state == DQNDashboard.MENU:
                    self._on_menu_key(event.key)
                else:
                    self._on_run_key(event.key)

    # --------------------------- Menue -------------------------------- #
    def _menu_entries(self) -> list[tuple[str, str | None, str]]:
        """(Beschriftung, angezeigter Wert oder None fuer Buttons, Schluessel)."""
        entries: list[tuple[str, str | None, str]] = []
        if self.trainer is not None:
            entries.append(("◀ WEITER TRAINIEREN", None, "resume"))
        entries += [
            ("Spiele gleichzeitig", str(NUM_GAMES_OPTIONS[self.games_idx]), "games"),
            ("Netzgröße", HIDDEN_PRESETS[self.hidden_idx][0], "hidden"),
            ("Lernrate", LR_PRESETS[self.lr_idx][0], "lr"),
            ("Neugier klingt ab über", EPS_PRESETS[self.eps_idx][0], "eps"),
            ("Weitsicht (gamma)", GAMMA_PRESETS[self.gamma_idx][0], "gamma"),
            ("Früchte", str(FRUIT_OPTIONS[self.fruit_idx]), "fruit"),
            ("NEUES TRAINING STARTEN", None, "start"),
        ]
        return entries

    def _on_menu_key(self, key: int) -> None:
        entries = self._menu_entries()
        if key in (pygame.K_UP, pygame.K_w):
            self.menu_row = (self.menu_row - 1) % len(entries)
        elif key in (pygame.K_DOWN, pygame.K_s):
            self.menu_row = (self.menu_row + 1) % len(entries)
        elif key in (pygame.K_LEFT, pygame.K_a):
            self._menu_adjust(entries[self.menu_row][2], -1)
        elif key in (pygame.K_RIGHT, pygame.K_d):
            self._menu_adjust(entries[self.menu_row][2], +1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            kind = entries[self.menu_row][2]
            if kind == "resume":
                self.state = DQNDashboard.RUNNING
            elif kind == "start":
                self._start_training()
            else:
                self._menu_adjust(kind, +1)
        elif key == pygame.K_ESCAPE:
            self.running = False

    def _menu_adjust(self, kind: str, delta: int) -> None:
        if kind == "games":
            self.games_idx = (self.games_idx + delta) % len(NUM_GAMES_OPTIONS)
        elif kind == "hidden":
            self.hidden_idx = (self.hidden_idx + delta) % len(HIDDEN_PRESETS)
        elif kind == "lr":
            self.lr_idx = (self.lr_idx + delta) % len(LR_PRESETS)
        elif kind == "eps":
            self.eps_idx = (self.eps_idx + delta) % len(EPS_PRESETS)
        elif kind == "gamma":
            self.gamma_idx = (self.gamma_idx + delta) % len(GAMMA_PRESETS)
        elif kind == "fruit":
            self.fruit_idx = (self.fruit_idx + delta) % len(FRUIT_OPTIONS)

    # ------------------------- Laufendes Training --------------------- #
    def _on_run_key(self, key: int) -> None:
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_ESCAPE:
            self.state = DQNDashboard.MENU
            self.menu_row = 0
        elif key in (pygame.K_RIGHT, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self.speed_idx = min(len(SPEED_LEVELS) - 1, self.speed_idx + 1)
        elif key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_idx = max(0, self.speed_idx - 1)
        elif key == pygame.K_t:
            self.turbo = not self.turbo

    # ================================================================== #
    # Training starten
    # ================================================================== #
    def _start_training(self) -> None:
        base = self.base_cfg
        self.cfg = DQNConfig(
            grid_cols=base.grid_cols,
            grid_rows=base.grid_rows,
            fruit_count=FRUIT_OPTIONS[self.fruit_idx],
            wrap_walls=base.wrap_walls,
            hidden=HIDDEN_PRESETS[self.hidden_idx][1],
            num_games=NUM_GAMES_OPTIONS[self.games_idx],
            learning_rate=LR_PRESETS[self.lr_idx][1],
            gamma=GAMMA_PRESETS[self.gamma_idx][1],
            batch_size=base.batch_size,
            buffer_size=base.buffer_size,
            min_buffer=base.min_buffer,
            train_iters_per_step=base.train_iters_per_step,
            target_update=base.target_update,
            grad_clip=base.grad_clip,
            double_dqn=base.double_dqn,
            eps_start=base.eps_start,
            eps_end=base.eps_end,
            eps_decay_steps=EPS_PRESETS[self.eps_idx][1],
            reward_fruit=base.reward_fruit,
            reward_death=base.reward_death,
            reward_step=base.reward_step,
            reward_closer=base.reward_closer,
            reward_farther=base.reward_farther,
            starve_base=base.starve_base,
            starve_growth=base.starve_growth,
        )
        self.trainer = MultiGameTrainer(self.cfg, log_to_csv=True)
        self.paused = False
        self.turbo = False
        self.speed_idx = DEFAULT_SPEED
        self._perf.clear()
        self.state = DQNDashboard.RUNNING

    # ================================================================== #
    # Training vorantreiben
    # ================================================================== #
    def _update(self) -> None:
        if self.trainer is None or self.paused:
            return

        if self.turbo:
            # Turbo: so viele Ticks wie in ein kleines Zeitbudget passen. Das
            # Budget sorgt dafuer, dass das Fenster trotzdem auf Tasten reagiert.
            t_end = time.perf_counter() + TURBO_BUDGET
            while time.perf_counter() < t_end:
                for _ in range(16):   # in Bloecken, damit die Uhr nicht bremst
                    self.trainer.step()
        else:
            for _ in range(SPEED_LEVELS[self.speed_idx]):
                self.trainer.step()

        self._perf.append((time.perf_counter(), self.trainer.total_moves))

    def _moves_per_second(self) -> float:
        """Durchsatz aus dem gleitenden Messfenster."""
        if len(self._perf) < 2:
            return 0.0
        (t0, m0), (t1, m1) = self._perf[0], self._perf[-1]
        dt = t1 - t0
        return (m1 - m0) / dt if dt > 0 else 0.0

    # ================================================================== #
    # Zeichnen: Menue
    # ================================================================== #
    def _draw_menu(self) -> None:
        self.screen.fill(Palette.BG)
        cx = WIN_W // 2
        self._center(self.f_title, "Deep Q-Learning — Training", Palette.ACCENT, cx, 84)
        self._center(self.f_small,
                     "Ein Gehirn, ein Erfahrungs-Tagebuch, mehrere Schlangen gleichzeitig",
                     Palette.TEXT_DIM, cx, 120)

        entries = self._menu_entries()
        px, pw, row_h, gap, y0 = 350, 620, 50, 11, 176
        for i, (label, value, _kind) in enumerate(entries):
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

        note = ("Feineinstellungen (Belohnungen, Puffer, Batch, Verhungern-Limit): "
                "ai/dqn/config.py")
        self._center(self.f_tiny, note, Palette.TEXT_DIM, cx, WIN_H - 54)
        self._center(self.f_tiny,
                     "Pfeil hoch/runter wählen   ·   links/rechts ändern   ·   Enter startet",
                     Palette.TEXT_DIM, cx, WIN_H - 30)

    # ================================================================== #
    # Zeichnen: laufendes Training
    # ================================================================== #
    def _draw_running(self) -> None:
        self.screen.fill(Palette.BG)
        stats = self.trainer.stats() if self.trainer else DQNStats()
        self._draw_header(stats)

        content = pygame.Rect(PAD, HEADER_H + PAD, WIN_W - 2 * PAD, WIN_H - HEADER_H - 2 * PAD)

        if self.turbo:
            self._draw_turbo_placeholder(content)
        else:
            self._draw_boards(content)

        below = pygame.Rect(content.x, content.y + BOARD_AREA_H + 6,
                            content.width, content.bottom - content.y - BOARD_AREA_H - 6)
        self._draw_stat_tiles(below, stats)
        self._draw_death_bars(below, stats)
        self._draw_curve(below, stats)

    def _draw_header(self, s: DQNStats) -> None:
        pygame.draw.rect(self.screen, Palette.HEADER_BG, pygame.Rect(0, 0, WIN_W, HEADER_H))
        pygame.draw.line(self.screen, Palette.BORDER, (0, HEADER_H - 1), (WIN_W, HEADER_H - 1), 2)

        title = self.f_h2.render(f"Episode {s.total_episodes}", Palette.TEXT)
        self.screen.blit(title, (PAD, 18))

        if self.paused:
            status, col = "PAUSE", Palette.ACCENT_WARN
        elif self.turbo:
            status, col = "TURBO", TURBO_COLOR
        else:
            status, col = f"LÄUFT   ·   {SPEED_LEVELS[self.speed_idx]}x", Palette.ACCENT
        status += f"   ·   {self._moves_per_second():,.0f} Züge/s".replace(",", ".")
        ss = self.f_body.render(status, col)
        self.screen.blit(ss, ss.get_rect(midright=(WIN_W - PAD, HEADER_H // 2)))

        hint = self.f_tiny.render(
            "Leertaste Pause   ·   ←/→ Geschwindigkeit   ·   T Turbo   ·   Esc Menü",
            Palette.TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, HEADER_H // 2)))

    # --------------------------- Spielfelder --------------------------- #
    def _board_geometry(self) -> tuple[int, int, int, int, int]:
        """(cols, rows, cell, board_w, board_h) fuer das Feld-Raster."""
        n = len(self.trainer.games)
        cols = min(n, 5)
        rows = math.ceil(n / cols)
        gcols, grows = self.cfg.grid_cols, self.cfg.grid_rows

        avail_w = WIN_W - 2 * PAD - (cols - 1) * BOARD_GAP
        avail_h = BOARD_AREA_H - rows * CAPTION_H - (rows - 1) * BOARD_GAP
        cell = max(2, int(min(avail_w / cols / gcols, avail_h / rows / grows)))
        return cols, rows, cell, cell * gcols, cell * grows

    def _draw_boards(self, content: pygame.Rect) -> None:
        cols, rows, cell, bw, bh = self._board_geometry()
        total_w = cols * bw + (cols - 1) * BOARD_GAP
        x0 = content.x + (content.width - total_w) // 2

        for i, game in enumerate(self.trainer.games):
            r, c = divmod(i, cols)
            x = x0 + c * (bw + BOARD_GAP)
            y = content.y + r * (bh + CAPTION_H + BOARD_GAP)

            caption = f"Spiel {i + 1}   ·   Score {game.score}   ·   Länge {game.length}"
            self._text(self.f_tiny, caption, Palette.TEXT_DIM, x + 2, y + 2)

            board = pygame.Rect(x, y + CAPTION_H, bw, bh)
            pygame.draw.rect(self.screen, Palette.BOARD_A, board, border_radius=6)
            pygame.draw.rect(self.screen, Palette.BORDER, board, width=1, border_radius=6)

            # Frucht(e)
            for (fx, fy) in game.fruits:
                pygame.draw.rect(self.screen, Palette.FRUIT,
                                 (board.x + fx * cell, board.y + fy * cell, cell, cell))
            # Schlange: Kopf hell, Koerper dunkler
            for idx, (sx, sy) in enumerate(game.snake):
                color = Palette.SNAKE_HEAD if idx == 0 else Palette.SNAKE_BODY
                pygame.draw.rect(
                    self.screen, color,
                    (board.x + sx * cell, board.y + sy * cell, max(1, cell - 1), max(1, cell - 1)))

    def _draw_turbo_placeholder(self, content: pygame.Rect) -> None:
        rect = pygame.Rect(content.x, content.y, content.width, BOARD_AREA_H - 6)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
        self._center(self.f_title, "TURBO", TURBO_COLOR, rect.centerx, rect.centery - 22)
        self._center(self.f_small,
                     "Felder werden nicht gezeichnet — die volle Rechenzeit fließt ins Lernen.",
                     Palette.TEXT_DIM, rect.centerx, rect.centery + 14)
        self._center(self.f_tiny, "T drücken, um wieder zuzusehen",
                     Palette.TEXT_DIM, rect.centerx, rect.centery + 40)

    # ------------------------------ Kacheln ---------------------------- #
    def _draw_stat_tiles(self, area: pygame.Rect, s: DQNStats) -> None:
        eff = f"{s.mean_steps_per_fruit:.1f}" if s.mean_steps_per_fruit else "—"
        loss = f"{s.loss:.3f}" if s.loss is not None else "sammelt …"
        tiles = [
            ("Ø SCORE (LETZTE 100)", f"{s.mean_score:.2f}", Palette.ACCENT),
            ("BESTWERT", str(s.best_score), CURVE_BEST),
            ("NEUGIER ε", f"{s.epsilon:.3f}", Palette.TEXT),
            ("LOSS (LERNFEHLER)", loss, Palette.TEXT),
            ("TAGEBUCH GEFÜLLT", f"{s.buffer_fill * 100:.0f}%", Palette.TEXT),
            ("EPISODEN", f"{s.total_episodes}", Palette.TEXT),
            ("LERNSCHRITTE", f"{s.learn_steps}", Palette.TEXT),
            ("EFFIZIENZ (SCHR./FRUCHT)", eff, Palette.TEXT),
        ]
        cols, rows = 4, 2
        gap = 14
        tile_w = (area.width - (cols - 1) * gap) / cols
        tile_h = 84
        for i, (label, value, color) in enumerate(tiles):
            r, c = divmod(i, cols)
            rect = pygame.Rect(int(area.x + c * (tile_w + gap)),
                               int(area.y + r * (tile_h + gap)),
                               int(tile_w), tile_h)
            pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
            self._center(self.f_label, label, Palette.TEXT_DIM, rect.centerx, rect.y + 20)
            self._center(self.f_value, value, color, rect.centerx, rect.y + 54)

        self._tiles_bottom = area.y + rows * tile_h + (rows - 1) * gap

    # --------------------------- Todesursachen ------------------------- #
    def _draw_death_bars(self, area: pygame.Rect, s: DQNStats) -> None:
        y = self._tiles_bottom + 18
        self._text(self.f_label, "TODESURSACHEN (alle Episoden)", Palette.TEXT_DIM, area.x, y)
        y += 22
        total = max(1, sum(s.deaths.values()))
        causes = [
            ("Wand", s.deaths.get("wall", 0), (231, 111, 81)),
            ("Selbst", s.deaths.get("self", 0), (233, 196, 106)),
            ("Verhungert", s.deaths.get("starvation", 0), (109, 158, 235)),
            ("Gewonnen", s.deaths.get("won", 0), Palette.ACCENT),
        ]
        col_w = area.width / 2
        for i, (name, count, color) in enumerate(causes):
            r, c = divmod(i, 2)
            self._death_bar(int(area.x + c * col_w), y + r * 24,
                            int(col_w - 24), name, count, count / total, color)

        self._death_bottom = y + 2 * 24 + 8

    def _death_bar(self, x, y, w, name, count, frac, color) -> None:
        label = self.f_small.render(name, Palette.TEXT)
        self.screen.blit(label, (x, y))
        bar_x = x + 130
        bar_w = w - 130 - 60
        pygame.draw.rect(self.screen, (32, 36, 48), (bar_x, y + 2, bar_w, 12), border_radius=6)
        if frac > 0:
            pygame.draw.rect(self.screen, color,
                             (bar_x, y + 2, max(2, int(bar_w * frac)), 12), border_radius=6)
        cnt = self.f_small.render(f"{count}  ({frac * 100:.0f}%)", Palette.TEXT_DIM)
        self.screen.blit(cnt, cnt.get_rect(midright=(x + w, y + 8)))

    # ----------------------------- Lernkurve --------------------------- #
    def _draw_curve(self, area: pygame.Rect, s: DQNStats) -> None:
        rect = pygame.Rect(area.x, self._death_bottom, area.width,
                           area.bottom - self._death_bottom)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
        self._text(self.f_label, "LERNKURVE  (Ø Score der letzten 100 Partien)",
                   Palette.TEXT_DIM, rect.x + 12, rect.y + 10)

        plot = pygame.Rect(rect.x + 46, rect.y + 34, rect.width - 62, rect.height - 52)
        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.y), (plot.x, plot.bottom), 1)
        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.bottom), (plot.right, plot.bottom), 1)

        curve = s.curve
        if len(curve) < 2:
            self._text(self.f_tiny, "sammelt Daten …", Palette.TEXT_DIM, plot.x + 8, plot.y + 8)
            return

        vmax = max(1.0, max(curve))
        n = len(curve)
        # Bei sehr vielen Punkten nur jeden k-ten zeichnen -- sieht identisch aus,
        # spart aber Zeichenzeit.
        stride = max(1, n // max(1, plot.width))
        points = [
            (plot.x + (i / (n - 1)) * plot.width,
             plot.bottom - (curve[i] / vmax) * plot.height)
            for i in range(0, n, stride)
        ]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, CURVE_MEAN, False, points, 2)

        self._text(self.f_tiny, f"{vmax:.1f}", Palette.TEXT_DIM, rect.x + 10, plot.y - 4)
        self._text(self.f_tiny, "0", Palette.TEXT_DIM, rect.x + 10, plot.bottom - 10)
        self._text(self.f_tiny, f"{s.total_episodes} Episoden", Palette.TEXT_DIM,
                   plot.right - 100, plot.bottom + 6)

    # ------------------------------------------------------------------ #
    # kleine Text-Helfer
    # ------------------------------------------------------------------ #
    def _text(self, font, text, color, x, y) -> None:
        self.screen.blit(font.render(text, color), (x, y))

    def _center(self, font, text, color, cx, cy) -> None:
        surf = font.render(text, color)
        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))


def main(cfg: DQNConfig | None = None) -> None:
    DQNDashboard(cfg).run()


if __name__ == "__main__":
    main()
