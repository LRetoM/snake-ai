"""Der Trainer: EIN Gehirn, EIN Tagebuch, mehrere gleichzeitig spielende Schlangen.

Die Architektur in einem Bild
-----------------------------
Stell dir fuenf Fahrschueler vor, die gleichzeitig in fuenf Autos sitzen -- aber
sie teilen sich EIN Gehirn und EIN gemeinsames Fahrtenbuch. Jeder Schueler faehrt
seine eigene Strecke (eigenes Spielfeld, eigener Fruchtzufall), aber:

  - alle fragen dasselbe Gehirn, was sie als naechstes tun sollen,
  - alle schreiben ihre Erlebnisse ins gleiche Fahrtenbuch,
  - gelernt wird aus zufaellig gemischten Seiten dieses einen Fahrtenbuchs.

Faehrt Schlange 3 in die Wand, verbessert dieser Fehler das Gehirn -- und damit
sofort auch die Schlangen 1, 2, 4 und 5. So "lernen sie voneinander", ohne dass
sie sich je absprechen. Das ist genau die Idee hinter parallelem DQN (bekannt aus
"Ape-X" & Co.): mehr Erfahrung pro Sekunde und eine buntere Mischung im Tagebuch.

Alles laeuft in EINEM Prozess und EINEM Fenster -- bewusst kein Multiprocessing.
Das waere fuer den Python-Einstieg viel fehleranfaelliger und wuerde hier kaum
etwas bringen, weil unsere Netze winzig sind.

Was hier NICHT passiert
-----------------------
Kein pygame, keine Grafik. Der Trainer rechnet nur. Das Dashboard
(dashboard/dqn_view.py) ruft `step()` so oft auf, wie es gerade darf, und liest
dann `stats()` aus. Dadurch koennen wir die Geschwindigkeit voellig frei drehen
(inklusive Turbo), ohne dass die Lernlogik davon etwas mitbekommt.

Die Verhungern-Regel
--------------------
`starve_limit(laenge) = starve_base + starve_growth * laenge` -- laeuft eine
Schlange so viele Zuege ohne Frucht, brechen wir die Partie ab und behandeln das
wie einen Tod. Wichtig: Das ist eine TRAININGS-Regel, keine Spielregel. Sie lebt
deshalb hier und nicht in game/snake_game.py. Sie verhindert, dass die KI eine
"sichere Endlosschleife" als Lieblingsstrategie entdeckt (nie sterben, nie
punkten). Dass das Limit mit der Laenge waechst, ist nur fair: eine lange
Schlange braucht laenger, um sicher zur Frucht zu manoevrieren.
"""

from __future__ import annotations

import csv
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from game.config import GameConfig
from game.snake_game import Action, SnakeGame
from ai.perception import perceive
from ai.dqn.agent import DQNAgent
from ai.dqn.config import DQNConfig
from ai.dqn.memory import ReplayBuffer
from ai.dqn.reward import compute_reward, fruit_distance

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(_ROOT, "models")
LOG_DIR = os.path.join(_ROOT, "logs")
CHAMPION_PATH = os.path.join(MODEL_DIR, "dqn_champion.pt")


@dataclass
class DQNStats:
    """Momentaufnahme des Trainings -- alles, was das Dashboard anzeigt."""
    total_steps: int = 0          # Trainer-Ticks (ein Tick = alle Spiele je 1 Zug)
    total_moves: int = 0          # einzelne Schlangen-Zuege insgesamt
    total_episodes: int = 0       # abgeschlossene Partien insgesamt
    epsilon: float = 1.0          # aktuelle Neugier
    mean_score: float = 0.0       # gleitendes Mittel der letzten Partien
    best_score: int = 0           # bester je erreichter Score
    mean_steps_per_fruit: float | None = None   # Effizienz (kleiner = besser)
    loss: float | None = None     # geglaetteter Lernfehler
    mean_q: float = 0.0           # durchschnittliche Bewertung des besten Zuges
    buffer_fill: float = 0.0      # Tagebuch-Fuellstand 0..1
    buffer_len: int = 0
    learn_steps: int = 0
    deaths: dict = field(default_factory=dict)
    curve: list = field(default_factory=list)   # Verlauf des Ø-Scores


class MultiGameTrainer:
    """Treibt `num_games` Spiele parallel an und trainiert EIN gemeinsames Netz."""

    def __init__(self, cfg: DQNConfig | None = None, seed: int | None = None,
                 log_to_csv: bool = True) -> None:
        self.cfg = cfg or DQNConfig()
        cfg = self.cfg

        # Ein Basis-Seed sorgt dafuer, dass jedes Spiel seinen EIGENEN Zufall
        # bekommt (sonst spawnen alle Fruechte identisch -> die 5 Spiele waeren
        # Klone und wuerden das Tagebuch mit lauter gleichen Erfahrungen fuellen).
        base_seed = seed if seed is not None else random.randrange(1 << 30)

        self.agent = DQNAgent(cfg, seed=base_seed)
        self.buffer = ReplayBuffer(cfg.buffer_size,
                                   rng=np.random.default_rng(base_seed + 999))

        game_cfg = GameConfig(
            grid_cols=cfg.grid_cols,
            grid_rows=cfg.grid_rows,
            fruit_count=cfg.fruit_count,
            wrap_walls=cfg.wrap_walls,
        )
        self.games: list[SnakeGame] = [
            SnakeGame(game_cfg, rng=random.Random(base_seed + i))
            for i in range(cfg.num_games)
        ]

        # Aktueller Wahrnehmungsvektor + Fruchtabstand pro Spiel. Beides fuehren
        # wir mit, damit wir es nicht jeden Zug doppelt ausrechnen muessen.
        self.states = np.stack([perceive(g) for g in self.games])
        self.dists = [fruit_distance(g) for g in self.games]
        self.episode_steps = [0] * cfg.num_games

        # ---- Statistik ------------------------------------------------- #
        self.epsilon = cfg.eps_start
        self.total_steps = 0
        self.total_moves = 0
        self.total_episodes = 0
        self.best_score = 0
        self.recent_scores: deque[int] = deque(maxlen=100)
        self.recent_efficiency: deque[float] = deque(maxlen=100)
        self.deaths = {"wall": 0, "self": 0, "starvation": 0, "won": 0}
        self.loss_smoothed: float | None = None
        self.score_curve: list[float] = []
        self._curve_stride = 1
        self.champion_path: str | None = None

        # ---- CSV-Protokoll (optional, landet in logs/) ------------------ #
        self._csv_path: str | None = None
        if log_to_csv:
            os.makedirs(LOG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._csv_path = os.path.join(LOG_DIR, f"dqn-{stamp}.csv")
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(
                    ["episode", "steps", "epsilon", "mean_score", "best_score", "loss"]
                )

    # ================================================================== #
    # Ein Trainings-Tick: alle Spiele machen EINEN Zug, dann wird gelernt
    # ================================================================== #
    def step(self) -> None:
        cfg = self.cfg

        # 1) EIN Netz-Durchlauf fuer alle Spiele gleichzeitig (epsilon-greedy).
        actions = self.agent.act_batch(self.states, self.epsilon)

        for i, game in enumerate(self.games):
            action_idx = int(actions[i])
            result = game.step_action(Action(action_idx))
            self.episode_steps[i] += 1

            died = not result.alive

            # 2) Verhungern pruefen (Trainings-Regel, siehe Modul-Docstring).
            if not died and not result.won:
                if game.steps_since_fruit >= cfg.starve_limit(game.length):
                    died = True
                    game.death_cause = "starvation"

            terminal = died or result.won

            # 3) Folge-Wahrnehmung + neuen Fruchtabstand bestimmen.
            #    Bei einem Ende brauchen wir beides nicht: das Lernziel blendet
            #    die Zukunft dort ohnehin aus (der (1-done)-Faktor in agent.py).
            #    Ausserdem waere perceive() nach einem Sieg gar nicht moeglich --
            #    dann liegt keine Frucht mehr auf dem Feld.
            if terminal:
                next_state = self.states[i]
                dist_after = self.dists[i]
            else:
                next_state = perceive(game)
                dist_after = fruit_distance(game)

            # 4) Belohnung berechnen (reines Feedback, siehe reward.py).
            reward = compute_reward(cfg, result.ate_fruit, died,
                                    self.dists[i], dist_after)

            # 5) Erfahrung ins gemeinsame Tagebuch legen.
            self.buffer.push(self.states[i], action_idx, reward, next_state, terminal)

            # 6) Weiterschalten -- entweder naechste Situation oder neue Partie.
            if terminal:
                self._finish_episode(i, game, won=result.won)
            else:
                self.states[i] = next_state
                self.dists[i] = dist_after

        self.total_steps += 1
        self.total_moves += len(self.games)

        # 7) Aus dem Tagebuch lernen (macht nichts, solange zu wenig drin ist).
        loss = self.agent.learn(self.buffer)
        if loss is not None:
            # Exponentielle Glaettung: der angezeigte Loss zappelt sonst so
            # stark, dass man den Trend nicht sieht.
            self.loss_smoothed = loss if self.loss_smoothed is None \
                else 0.99 * self.loss_smoothed + 0.01 * loss

        # 8) Neugier linear abklingen lassen: viel ausprobieren -> vertrauen.
        progress = min(1.0, self.total_steps / max(1, cfg.eps_decay_steps))
        self.epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * progress

    # ------------------------------------------------------------------ #
    # Episode abschliessen (Statistik + Neustart dieses einen Spiels)
    # ------------------------------------------------------------------ #
    def _finish_episode(self, i: int, game: SnakeGame, won: bool) -> None:
        score = game.score
        steps = self.episode_steps[i]

        self.total_episodes += 1
        self.recent_scores.append(score)
        if score > 0:
            self.recent_efficiency.append(steps / score)

        cause = "won" if won else (game.death_cause or "wall")
        self.deaths[cause] = self.deaths.get(cause, 0) + 1

        # Neuen Rekord? -> Gehirn wegspeichern (models/ ist gitignored).
        if score > self.best_score:
            self.best_score = score
            self._save_champion(score)

        # Lernkurven-Punkt anhaengen (gleitendes Mittel der letzten Partien).
        if self.total_episodes % self._curve_stride == 0:
            self.score_curve.append(self.mean_score())
            # Damit die Kurve nach zehntausenden Episoden nicht ins Unendliche
            # waechst: bei 2000 Punkten jeden zweiten wegwerfen und ab dann nur
            # noch halb so oft aufzeichnen (der Verlauf bleibt derselbe).
            if len(self.score_curve) > 2000:
                self.score_curve = self.score_curve[::2]
                self._curve_stride *= 2

        self._log_row()

        # Frisches Spiel fuer diesen Slot.
        game.reset()
        self.states[i] = perceive(game)
        self.dists[i] = fruit_distance(game)
        self.episode_steps[i] = 0

    # ------------------------------------------------------------------ #
    # Speichern & Protokoll
    # ------------------------------------------------------------------ #
    def _save_champion(self, score: int) -> None:
        cfg = self.cfg
        self.champion_path = self.agent.save_checkpoint(
            CHAMPION_PATH,
            {
                "score": score,
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
                "grid_cols": cfg.grid_cols,
                "grid_rows": cfg.grid_rows,
                "fruit_count": cfg.fruit_count,
                "wrap_walls": cfg.wrap_walls,
            },
        )

    def _log_row(self) -> None:
        """Schreibt alle 25 Episoden eine Zeile ins CSV-Protokoll."""
        if not self._csv_path or self.total_episodes % 25 != 0:
            return
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                self.total_episodes,
                self.total_steps,
                f"{self.epsilon:.4f}",
                f"{self.mean_score():.3f}",
                self.best_score,
                "" if self.loss_smoothed is None else f"{self.loss_smoothed:.5f}",
            ])

    # ------------------------------------------------------------------ #
    # Auswertung
    # ------------------------------------------------------------------ #
    def mean_score(self) -> float:
        """Ø-Score der letzten (bis zu) 100 Partien."""
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def stats(self) -> DQNStats:
        """Momentaufnahme fuer das Dashboard."""
        eff = None
        if self.recent_efficiency:
            eff = sum(self.recent_efficiency) / len(self.recent_efficiency)
        return DQNStats(
            total_steps=self.total_steps,
            total_moves=self.total_moves,
            total_episodes=self.total_episodes,
            epsilon=self.epsilon,
            mean_score=self.mean_score(),
            best_score=self.best_score,
            mean_steps_per_fruit=eff,
            loss=self.loss_smoothed,
            mean_q=self.agent.last_mean_q,
            buffer_fill=self.buffer.fill_ratio,
            buffer_len=len(self.buffer),
            learn_steps=self.agent.learn_steps,
            deaths=dict(self.deaths),
            curve=self.score_curve,
        )
