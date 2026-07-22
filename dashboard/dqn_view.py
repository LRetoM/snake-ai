"""Live-Fenster fuer das DQN-Training: Spiele + Kennzahlen + Lernkurven.

Anders als beim Neuroevolution-Dashboard (dashboard/live_view.py) siehst du hier
die Schlangen WIRKLICH spielen, waehrend sie lernen. Das ist bei DQN sinnvoll,
weil ein einziges Gehirn dauerhaft weitertrainiert -- man sieht also live, wie
aus zappeligem Unsinn allmaehlich zielgerichtetes Spiel wird.

Die wichtigste Zahl im Fenster ist PRUEFUNG: der Ø-Score aus Partien ganz ohne
Zufallszuege. Der Trainings-Score daneben ist immer etwas schlechter, weil die
KI beim Training absichtlich noch herumprobiert. Zum Vergleichen und Angeben
zaehlt die Pruefung -- das ist auch der Wert, den du beim Zuschauen siehst.

Aufbau des Fensters:
- oben: die parallel laufenden Spielfelder (klein, aber echt -- dieselbe Engine)
- Mitte: acht Kennzahlen-Kacheln + eine Zeile mit Detailwerten
- darunter: Todesursachen als Balken (gesamt und "zuletzt")
- unten: LERNKURVEN -- Trainings-Ø und Pruefung uebereinander, selbst gezeichnet
  mit pygame.draw.lines (bewusst ohne matplotlib).

Steuerung:
    Leertaste  Pause / weiter
    ←/→ oder +/-   Geschwindigkeit (Ticks pro gezeichnetem Bild)
    T          TURBO an/aus  (rechnet mit voller Kraft, zeichnet keine Felder)
    P          sofort eine Pruefung laufen lassen
    Esc        zurueck ins Menue (das Training bleibt erhalten)

Warum Geschwindigkeit und Zeichnen getrennt sind: Zeichnen kostet Zeit, die nicht
ins Lernen fliesst. Im Turbo-Modus rechnen wir ein festes Zeitbudget lang so viel
wie moeglich und zeigen nur noch die Zahlen -- der schnellste Lernmodus.

Wichtig fuer Python 3.14: Schriften kommen NUR ueber game.fonts.load_font --
pygame.font stuerzt dort beim Import ab (bekannter pygame-Bug).
"""

from __future__ import annotations

import math
import os
import time
from collections import deque

import pygame

from game.config import Palette
from game.fonts import load_font
from ai.dqn.config import DQNConfig
from ai.dqn.trainer import CHAMPION_PATH, DQNStats, MultiGameTrainer

# ----------------------------- Fenster-Layout ------------------------------ #
WIN_W, WIN_H = 1360, 900
HEADER_H = 60
PAD = 20
BOARD_AREA_H = 232
BOARD_GAP = 12
CAPTION_H = 16
MAX_DRAWN_BOARDS = 10

# ------------------------- Einstellbare Presets ---------------------------- #
# Nur die MENUE-Auswahl. Die volle Wahrheit steht in ai/dqn/config.py.
NUM_GAMES_OPTIONS = [1, 2, 4, 5, 6, 8, 12, 16]
PERCEPTION_PRESETS = [
    ("Reich (39 Werte)", "rich"),
    ("Einfach (11 Werte)", "simple"),
]
HIDDEN_PRESETS = [
    ("Klein (128, 128)", (128, 128)),
    ("Mittel (256, 128)", (256, 128)),
    ("Groß (256, 256)", (256, 256)),
    ("Sehr groß (512, 256)", (512, 256)),
]
LR_PRESETS = [
    ("Vorsichtig (0.0005)", 5e-4),
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
    ("Kurzsichtig (0.90)", 0.90),
    ("Normal (0.95)", 0.95),
    ("Weitsichtig (0.97)", 0.97),
]
NSTEP_PRESETS = [("1 Zug", 1), ("3 Züge", 3), ("5 Züge", 5)]
FRUIT_OPTIONS = list(range(1, 11))

# Wie viele Trainings-Ticks pro gezeichnetem Bild gerechnet werden.
SPEED_LEVELS = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_SPEED = 2
TURBO_BUDGET = 0.08      # Sekunden Rechenzeit pro Bild im Turbo-Modus

# ----------------------------- Extra-Farben -------------------------------- #
PANEL_BG = (22, 25, 35)
CURVE_TRAIN = (90, 120, 180)
CURVE_EVAL = Palette.ACCENT
GOLD = (250, 200, 90)
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
        self.f_label = load_font(12, "semibold")
        self.f_value = load_font(27, "semibold")
        self.f_body = load_font(16, "regular")
        self.f_small = load_font(14, "regular")
        self.f_tiny = load_font(11, "regular")

        self.base_cfg = cfg or DQNConfig()

        # Menue-Auswahl mit den Werten aus der Config als Startpunkt.
        self.games_idx = _index_of(NUM_GAMES_OPTIONS, self.base_cfg.num_games, 5)
        self.perc_idx = _preset_index(PERCEPTION_PRESETS, self.base_cfg.perception, 0)
        self.hidden_idx = _preset_index(HIDDEN_PRESETS, tuple(self.base_cfg.hidden), 1)
        self.lr_idx = _preset_index(LR_PRESETS, self.base_cfg.learning_rate, 1)
        self.eps_idx = _preset_index(EPS_PRESETS, self.base_cfg.eps_decay_steps, 1)
        self.gamma_idx = _preset_index(GAMMA_PRESETS, self.base_cfg.gamma, 1)
        self.nstep_idx = _preset_index(NSTEP_PRESETS, self.base_cfg.n_step, 1)
        self.fruit_idx = self.base_cfg.fruit_count - 1
        self.per_on = self.base_cfg.prioritized
        self.resume_on = False
        self.menu_row = 0
        self.menu_error: str | None = None

        self.state = DQNDashboard.MENU
        self.running = True

        self.trainer: MultiGameTrainer | None = None
        self.cfg: DQNConfig | None = None
        self.paused = False
        self.turbo = False
        self.speed_idx = DEFAULT_SPEED
        self._perf: deque = deque(maxlen=40)

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
        entries: list[tuple[str, str | None, str]] = []
        if self.trainer is not None:
            entries.append(("◀ WEITER TRAINIEREN", None, "resume"))
        champ = "Ja" if self.resume_on else "Nein"
        if not os.path.exists(CHAMPION_PATH):
            champ = "— (noch kein Champion gespeichert)"
        entries += [
            ("Wahrnehmung", PERCEPTION_PRESETS[self.perc_idx][0], "perc"),
            ("Spiele gleichzeitig", str(NUM_GAMES_OPTIONS[self.games_idx]), "games"),
            ("Netzgröße", HIDDEN_PRESETS[self.hidden_idx][0], "hidden"),
            ("Lernrate", LR_PRESETS[self.lr_idx][0], "lr"),
            ("Neugier klingt ab über", EPS_PRESETS[self.eps_idx][0], "eps"),
            ("Weitsicht (gamma)", GAMMA_PRESETS[self.gamma_idx][0], "gamma"),
            ("Erfahrungs-Ketten (n-Schritt)", NSTEP_PRESETS[self.nstep_idx][0], "nstep"),
            ("Tagebuch priorisieren", "An" if self.per_on else "Aus", "per"),
            ("Früchte", str(FRUIT_OPTIONS[self.fruit_idx]), "fruit"),
            ("Gespeicherten Champion weitertrainieren", champ, "cont"),
            ("NEUES TRAINING STARTEN", None, "start"),
        ]
        return entries

    def _on_menu_key(self, key: int) -> None:
        entries = self._menu_entries()
        self.menu_row = min(self.menu_row, len(entries) - 1)
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
        self.menu_error = None
        if kind == "games":
            self.games_idx = (self.games_idx + delta) % len(NUM_GAMES_OPTIONS)
        elif kind == "perc":
            self.perc_idx = (self.perc_idx + delta) % len(PERCEPTION_PRESETS)
        elif kind == "hidden":
            self.hidden_idx = (self.hidden_idx + delta) % len(HIDDEN_PRESETS)
        elif kind == "lr":
            self.lr_idx = (self.lr_idx + delta) % len(LR_PRESETS)
        elif kind == "eps":
            self.eps_idx = (self.eps_idx + delta) % len(EPS_PRESETS)
        elif kind == "gamma":
            self.gamma_idx = (self.gamma_idx + delta) % len(GAMMA_PRESETS)
        elif kind == "nstep":
            self.nstep_idx = (self.nstep_idx + delta) % len(NSTEP_PRESETS)
        elif kind == "per":
            self.per_on = not self.per_on
        elif kind == "fruit":
            self.fruit_idx = (self.fruit_idx + delta) % len(FRUIT_OPTIONS)
        elif kind == "cont":
            self.resume_on = not self.resume_on and os.path.exists(CHAMPION_PATH)

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
        elif key == pygame.K_p and self.trainer is not None:
            self.trainer.run_evaluation()

    # ================================================================== #
    # Training starten
    # ================================================================== #
    def _start_training(self) -> None:
        base = self.base_cfg
        cfg = DQNConfig(**vars(base))       # alle Feineinstellungen uebernehmen
        cfg.perception = PERCEPTION_PRESETS[self.perc_idx][1]
        cfg.num_games = NUM_GAMES_OPTIONS[self.games_idx]
        cfg.hidden = HIDDEN_PRESETS[self.hidden_idx][1]
        cfg.learning_rate = LR_PRESETS[self.lr_idx][1]
        cfg.eps_decay_steps = EPS_PRESETS[self.eps_idx][1]
        cfg.gamma = GAMMA_PRESETS[self.gamma_idx][1]
        cfg.n_step = NSTEP_PRESETS[self.nstep_idx][1]
        cfg.prioritized = self.per_on
        cfg.fruit_count = FRUIT_OPTIONS[self.fruit_idx]

        resume = CHAMPION_PATH if self.resume_on else None
        try:
            self.trainer = MultiGameTrainer(cfg, log_to_csv=True, resume_from=resume)
        except (ValueError, OSError) as exc:
            # Typischer Fall: der gespeicherte Champion passt nicht zur jetzt
            # eingestellten Wahrnehmung/Netzgroesse. Klar sagen statt abstuerzen.
            self.menu_error = str(exc)
            return

        self.cfg = cfg
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
            # So viele Ticks wie in ein kleines Zeitbudget passen. Das Budget
            # sorgt dafuer, dass das Fenster auf Tasten reagiert.
            t_end = time.perf_counter() + TURBO_BUDGET
            while time.perf_counter() < t_end:
                for _ in range(16):   # in Bloecken, damit die Uhr nicht bremst
                    self.trainer.step()
        else:
            for _ in range(SPEED_LEVELS[self.speed_idx]):
                self.trainer.step()

        self._perf.append((time.perf_counter(), self.trainer.total_moves))

    def _moves_per_second(self) -> float:
        if len(self._perf) < 2:
            return 0.0
        (t0, m0), (t1, m1) = self._perf[0], self._perf[-1]
        return (m1 - m0) / (t1 - t0) if t1 > t0 else 0.0

    # ================================================================== #
    # Zeichnen: Menue
    # ================================================================== #
    def _draw_menu(self) -> None:
        self.screen.fill(Palette.BG)
        cx = WIN_W // 2
        self._center(self.f_title, "Deep Q-Learning — Training", Palette.ACCENT, cx, 62)
        self._center(self.f_small,
                     "Ein Gehirn, ein Erfahrungs-Tagebuch, mehrere Schlangen gleichzeitig",
                     Palette.TEXT_DIM, cx, 96)

        entries = self._menu_entries()
        px, pw, row_h, gap, y0 = 330, 700, 42, 8, 130
        for i, (label, value, _kind) in enumerate(entries):
            y = y0 + i * (row_h + gap)
            selected = (i == self.menu_row)
            rect = pygame.Rect(px, y, pw, row_h)
            if value is None:
                bg = Palette.ACCENT if selected else (30, 35, 50)
                pygame.draw.rect(self.screen, bg, rect, border_radius=10)
                if not selected:
                    pygame.draw.rect(self.screen, Palette.BORDER, rect, width=2, border_radius=10)
                self._center(self.f_h2, label, Palette.BG if selected else Palette.TEXT,
                             rect.centerx, rect.centery)
            else:
                if selected:
                    pygame.draw.rect(self.screen, (26, 38, 34), rect, border_radius=10)
                    pygame.draw.rect(self.screen, Palette.ACCENT, rect, width=2, border_radius=10)
                lc = Palette.TEXT if selected else Palette.TEXT_DIM
                vc = Palette.ACCENT if selected else Palette.TEXT
                ls = self.f_body.render(label, lc)
                self.screen.blit(ls, ls.get_rect(midleft=(px + 20, rect.centery)))
                vs = self.f_body.render(value, vc)
                self.screen.blit(vs, vs.get_rect(midright=(px + pw - 20, rect.centery)))

        y = y0 + len(entries) * (row_h + gap) + 12
        if self.menu_error:
            self._center(self.f_small, self.menu_error, Palette.ACCENT_WARN, cx, y)
            y += 24
        self._center(self.f_tiny,
                     "Feineinstellungen (Belohnungen, Puffergröße, Batch, Verhungern-Limit): "
                     "ai/dqn/config.py", Palette.TEXT_DIM, cx, WIN_H - 46)
        self._center(self.f_tiny,
                     "Pfeil hoch/runter wählen   ·   links/rechts ändern   ·   Enter startet",
                     Palette.TEXT_DIM, cx, WIN_H - 26)

    # ================================================================== #
    # Zeichnen: laufendes Training
    # ================================================================== #
    def _draw_running(self) -> None:
        self.screen.fill(Palette.BG)
        stats = self.trainer.stats() if self.trainer else DQNStats()
        self._draw_header(stats)

        content = pygame.Rect(PAD, HEADER_H + 16, WIN_W - 2 * PAD,
                              WIN_H - HEADER_H - 16 - PAD)
        if self.turbo:
            self._draw_turbo_placeholder(content)
        else:
            self._draw_boards(content)

        below = pygame.Rect(content.x, content.y + BOARD_AREA_H + 8,
                            content.width, content.bottom - content.y - BOARD_AREA_H - 8)
        self._draw_stat_tiles(below, stats)
        self._draw_info_line(below, stats)
        self._draw_death_bars(below, stats)
        self._draw_curves(below, stats)

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
            status, col = f"LÄUFT · {SPEED_LEVELS[self.speed_idx]}x", Palette.ACCENT
        status += f"   ·   {self._moves_per_second():,.0f} Züge/s".replace(",", ".")
        ss = self.f_body.render(status, col)
        self.screen.blit(ss, ss.get_rect(midright=(WIN_W - PAD, HEADER_H // 2)))

        hint = self.f_tiny.render(
            "Leertaste Pause · ←/→ Tempo · T Turbo · P Prüfung · Esc Menü",
            Palette.TEXT_DIM)
        self.screen.blit(hint, hint.get_rect(center=(WIN_W // 2, HEADER_H // 2)))

    # --------------------------- Spielfelder --------------------------- #
    def _draw_boards(self, content: pygame.Rect) -> None:
        games = self.trainer.games[:MAX_DRAWN_BOARDS]
        n = len(games)
        cols = min(n, 5)
        rows = math.ceil(n / cols)
        gcols, grows = self.cfg.grid_cols, self.cfg.grid_rows

        avail_w = content.width - (cols - 1) * BOARD_GAP
        avail_h = BOARD_AREA_H - rows * CAPTION_H - (rows - 1) * BOARD_GAP
        cell = max(2, int(min(avail_w / cols / gcols, avail_h / rows / grows)))
        bw, bh = cell * gcols, cell * grows
        x0 = content.x + (content.width - (cols * bw + (cols - 1) * BOARD_GAP)) // 2

        for i, game in enumerate(games):
            r, c = divmod(i, cols)
            x = x0 + c * (bw + BOARD_GAP)
            y = content.y + r * (bh + CAPTION_H + BOARD_GAP)

            self._text(self.f_tiny, f"#{i + 1}  Score {game.score}  ·  Länge {game.length}",
                       Palette.TEXT_DIM, x + 2, y + 1)

            board = pygame.Rect(x, y + CAPTION_H, bw, bh)
            pygame.draw.rect(self.screen, Palette.BOARD_A, board, border_radius=5)
            pygame.draw.rect(self.screen, Palette.BORDER, board, width=1, border_radius=5)

            for (fx, fy) in game.fruits:
                pygame.draw.rect(self.screen, Palette.FRUIT,
                                 (board.x + fx * cell, board.y + fy * cell, cell, cell))
            size = max(1, cell - 1)
            for idx, (sx, sy) in enumerate(game.snake):
                color = Palette.SNAKE_HEAD if idx == 0 else Palette.SNAKE_BODY
                pygame.draw.rect(self.screen, color,
                                 (board.x + sx * cell, board.y + sy * cell, size, size))

    def _draw_turbo_placeholder(self, content: pygame.Rect) -> None:
        rect = pygame.Rect(content.x, content.y, content.width, BOARD_AREA_H - 6)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
        self._center(self.f_title, "TURBO", TURBO_COLOR, rect.centerx, rect.centery - 20)
        self._center(self.f_small,
                     "Felder werden nicht gezeichnet — die volle Rechenzeit fließt ins Lernen.",
                     Palette.TEXT_DIM, rect.centerx, rect.centery + 16)
        self._center(self.f_tiny, "T drücken, um wieder zuzusehen",
                     Palette.TEXT_DIM, rect.centerx, rect.centery + 42)

    # ------------------------------ Kacheln ---------------------------- #
    def _draw_stat_tiles(self, area: pygame.Rect, s: DQNStats) -> None:
        eff = f"{s.mean_steps_per_fruit:.1f}" if s.mean_steps_per_fruit else "—"
        loss = f"{s.loss:.3f}" if s.loss is not None else "sammelt …"
        pruefung = f"{s.eval_score:.1f}" if s.eval_score is not None else "…"
        tiles = [
            ("PRÜFUNG (OHNE ZUFALL)", pruefung, Palette.ACCENT),
            ("BESTE PRÜFUNG = CHAMPION", f"{s.eval_best:.1f}", GOLD),
            ("Ø SCORE IM TRAINING", f"{s.mean_score:.1f}", CURVE_TRAIN),
            ("BESTE EINZELPARTIE", str(max(s.best_score, s.eval_max)), Palette.TEXT),
            ("FELD GEFÜLLT", f"{s.fill_percent:.1f}%", Palette.TEXT),
            ("EFFIZIENZ (SCHR./FRUCHT)", eff, Palette.TEXT),
            ("NEUGIER ε", f"{s.epsilon:.3f}", Palette.TEXT),
            ("LOSS (LERNFEHLER)", loss, Palette.TEXT),
        ]
        cols, rows = 4, 2
        gap = 12
        tile_w = (area.width - (cols - 1) * gap) / cols
        tile_h = 78
        for i, (label, value, color) in enumerate(tiles):
            r, c = divmod(i, cols)
            rect = pygame.Rect(int(area.x + c * (tile_w + gap)),
                               int(area.y + r * (tile_h + gap)),
                               int(tile_w), tile_h)
            pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
            self._center(self.f_label, label, Palette.TEXT_DIM, rect.centerx, rect.y + 18)
            self._center(self.f_value, value, color, rect.centerx, rect.y + 50)
        self._tiles_bottom = area.y + rows * tile_h + (rows - 1) * gap

    def _draw_info_line(self, area: pygame.Rect, s: DQNStats) -> None:
        """Eine schmale Zeile mit den Detailwerten, die keine Kachel brauchen."""
        y = self._tiles_bottom + 10
        mins, secs = divmod(int(s.elapsed), 60)
        parts = [
            f"Median {s.median_score:.0f}",
            f"Ø Länge {s.mean_length:.1f}",
            f"Ø Züge/Partie {s.mean_steps:.0f}",
            f"Ø Q {s.mean_q:.2f}",
            f"Tagebuch {s.buffer_len:,}".replace(",", ".") + f" ({s.buffer_fill * 100:.0f}%)",
            f"Lernschritte {s.learn_steps:,}".replace(",", "."),
            f"Ticks {s.total_steps:,}".replace(",", "."),
            f"Laufzeit {mins}:{secs:02d}",
        ]
        self._text(self.f_small, "   ·   ".join(parts), Palette.TEXT_DIM, area.x, y)
        self._info_bottom = y + 22

    # --------------------------- Todesursachen ------------------------- #
    def _draw_death_bars(self, area: pygame.Rect, s: DQNStats) -> None:
        y = self._info_bottom + 6
        self._text(self.f_label, "TODESURSACHEN", Palette.TEXT_DIM, area.x, y)
        self._text(self.f_tiny, "links: alle Episoden   ·   rechts: die letzten 100",
                   Palette.TEXT_DIM, area.x + 130, y + 1)
        y += 20

        rows = [("Wand", "wall", (231, 111, 81)),
                ("Selbst", "self", (233, 196, 106)),
                ("Verhungert", "starvation", (109, 158, 235))]
        total = max(1, sum(s.deaths.values()))
        recent_total = max(1, sum(s.recent_deaths.values()))
        col_w = area.width / 2
        for i, (name, key, color) in enumerate(rows):
            yy = y + i * 20
            c = s.deaths.get(key, 0)
            self._death_bar(int(area.x), yy, int(col_w - 30), name, c, c / total, color)
            c = s.recent_deaths.get(key, 0)
            self._death_bar(int(area.x + col_w), yy, int(col_w - 30), name, c,
                            c / recent_total, color)
        self._death_bottom = y + 3 * 20 + 8

    def _death_bar(self, x, y, w, name, count, frac, color) -> None:
        self.screen.blit(self.f_small.render(name, Palette.TEXT), (x, y))
        bar_x = x + 100
        bar_w = w - 100 - 90
        pygame.draw.rect(self.screen, (32, 36, 48), (bar_x, y + 3, bar_w, 10), border_radius=5)
        if frac > 0:
            pygame.draw.rect(self.screen, color,
                             (bar_x, y + 3, max(2, int(bar_w * frac)), 10), border_radius=5)
        cnt = self.f_small.render(f"{count}  ({frac * 100:.0f}%)", Palette.TEXT_DIM)
        self.screen.blit(cnt, cnt.get_rect(midright=(x + w, y + 8)))

    # ----------------------------- Lernkurven -------------------------- #
    def _draw_curves(self, area: pygame.Rect, s: DQNStats) -> None:
        rect = pygame.Rect(area.x, self._death_bottom, area.width,
                           area.bottom - self._death_bottom)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=10)
        self._text(self.f_label, "LERNKURVE", Palette.TEXT_DIM, rect.x + 12, rect.y + 10)
        self._text(self.f_tiny, "Prüfung (ohne Zufall)", CURVE_EVAL, rect.x + 92, rect.y + 11)
        self._text(self.f_tiny, "Ø Training (mit Neugier)", CURVE_TRAIN, rect.x + 232, rect.y + 11)

        plot = pygame.Rect(rect.x + 46, rect.y + 32, rect.width - 62, rect.height - 52)
        pygame.draw.line(self.screen, Palette.BORDER, (plot.x, plot.y), (plot.x, plot.bottom), 1)
        pygame.draw.line(self.screen, Palette.BORDER,
                         (plot.x, plot.bottom), (plot.right, plot.bottom), 1)

        train, ev = s.curve, s.eval_curve
        if len(train) < 2:
            self._text(self.f_tiny, "sammelt Daten …", Palette.TEXT_DIM, plot.x + 8, plot.y + 8)
            return

        vmax = max(1.0, max(train), max(ev) if ev else 0.0)
        # Waagerechte Hilfslinien -- machen den Abstand zwischen den Kurven lesbar.
        for frac in (0.25, 0.5, 0.75):
            yy = plot.bottom - frac * plot.height
            pygame.draw.line(self.screen, (34, 38, 52), (plot.x, yy), (plot.right, yy), 1)
            self._text(self.f_tiny, f"{vmax * frac:.0f}", Palette.TEXT_DIM, rect.x + 12, yy - 6)

        n = len(train)
        stride = max(1, n // max(1, plot.width))
        pts = [(plot.x + (i / (n - 1)) * plot.width,
                plot.bottom - (train[i] / vmax) * plot.height)
               for i in range(0, n, stride)]
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, CURVE_TRAIN, False, pts, 2)

        # Pruefungspunkte liegen auf der Episoden-Achse -> passend einordnen.
        if len(ev) >= 2 and s.total_episodes > 0:
            epts = [(plot.x + min(1.0, e / s.total_episodes) * plot.width,
                     plot.bottom - (v / vmax) * plot.height)
                    for e, v in zip(s.eval_points, ev)]
            pygame.draw.lines(self.screen, CURVE_EVAL, False, epts, 2)
            pygame.draw.circle(self.screen, CURVE_EVAL, epts[-1], 3)

        self._text(self.f_tiny, f"{vmax:.0f}", Palette.TEXT_DIM, rect.x + 12, plot.y - 4)
        self._text(self.f_tiny, "0", Palette.TEXT_DIM, rect.x + 12, plot.bottom - 10)
        self._text(self.f_tiny, f"{s.total_episodes} Episoden", Palette.TEXT_DIM,
                   plot.right - 100, plot.bottom + 6)

    # ------------------------------------------------------------------ #
    def _text(self, font, text, color, x, y) -> None:
        self.screen.blit(font.render(text, color), (x, y))

    def _center(self, font, text, color, cx, cy) -> None:
        surf = font.render(text, color)
        self.screen.blit(surf, surf.get_rect(center=(cx, cy)))


# ----------------------------- kleine Helfer -------------------------------- #
def _index_of(options: list, value, fallback: int) -> int:
    return options.index(value) if value in options else fallback


def _preset_index(presets: list[tuple], value, fallback: int) -> int:
    for i, (_name, v) in enumerate(presets):
        if v == value:
            return i
    return fallback


def main(cfg: DQNConfig | None = None) -> None:
    DQNDashboard(cfg).run()


if __name__ == "__main__":
    main()
