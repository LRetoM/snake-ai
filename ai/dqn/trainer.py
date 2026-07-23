"""Der Trainer: EIN Gehirn, EIN Tagebuch, mehrere gleichzeitig spielende Schlangen.

Die Architektur in einem Bild
-----------------------------
Stell dir mehrere Fahrschueler vor, die gleichzeitig in mehreren Autos sitzen --
aber sie teilen sich EIN Gehirn und EIN gemeinsames Fahrtenbuch. Jeder faehrt
seine eigene Strecke (eigenes Spielfeld, eigener Fruchtzufall), aber:

  - alle fragen dasselbe Gehirn, was sie als naechstes tun sollen,
  - alle schreiben ihre Erlebnisse ins gleiche Fahrtenbuch,
  - gelernt wird aus zufaellig gemischten Seiten dieses einen Fahrtenbuchs.

Faehrt Schlange 3 in die Wand, verbessert dieser Fehler das Gehirn -- und damit
sofort auch alle anderen. So "lernen sie voneinander", ohne sich je abzusprechen.
Alles laeuft in EINEM Prozess und EINEM Fenster (bewusst kein Multiprocessing).

Der wichtigste Messwert: die PRUEFUNG
-------------------------------------
Der Ø-Score waehrend des Trainings ist systematisch zu niedrig -- die KI wuerfelt
dort ja noch bei jedem epsilon-ten Zug. Bei epsilon=0.01 und einer Partie ueber
400 Zuege sind das ~4 Zufallszuege, und ein einziger davon kann eine lange
Schlange umbringen. Deshalb spielt die KI regelmaessig ein paar Partien ganz
OHNE Zufall: das ist ihr echtes Koennen, und genau der Wert, den du beim
Zuschauen (watch_ai.py) siehst.

Der Champion wird ausdruecklich nach dem PRUEFUNGS-Durchschnitt gespeichert,
nicht nach einer einzelnen Gluecks-Partie. Ein Bot, der einmal zufaellig 91
geschafft hat, ist nicht besser als einer, der verlaesslich 40 schafft --
genau dieser Unterschied hat beim Neuroevolution-Champion fuer die
Enttaeuschung beim Zuschauen gesorgt.

Die Verhungern-Regel
--------------------
`starve_limit(laenge) = starve_base + starve_growth * laenge` -- laeuft eine
Schlange so viele Zuege ohne Frucht, brechen wir die Partie ab und behandeln das
wie einen Tod. Wichtig: Das ist eine TRAININGS-Regel, keine Spielregel. Sie lebt
deshalb hier und nicht in game/snake_game.py. Sie verhindert, dass die KI eine
"sichere Endlosschleife" als Lieblingsstrategie entdeckt.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field, fields

import numpy as np

from game.config import GameConfig
from game.snake_game import Action, SnakeGame
from ai.perception import get_perception
from ai.dqn.agent import DQNAgent
from ai.dqn.config import DQNConfig
from ai.dqn.memory import NStepChain, make_buffer
from ai.dqn.reward import compute_reward, fruit_distance

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(_ROOT, "models")
LOG_DIR = os.path.join(_ROOT, "logs")
# Legacy: Champion-Pfad von VOR der Brett-Umstellung (als es nur 20x20 gab).
# Nur noch als Lese-Fallback in resolve_champion_path() gebraucht -- neu
# gespeichert wird immer im brettspezifischen Schema unten.
_LEGACY_CHAMPION_PATH = os.path.join(MODEL_DIR, "dqn_champion.pt")


def champion_path(cols: int, rows: int) -> str:
    """Datei-Pfad fuer den Champion EINES bestimmten Bretts.

    Jede Brettgroesse bekommt ihre EIGENE Champion-Datei: ein Score 80 auf
    einem 9x9-Feld ist etwas ganz anderes als ein Score 80 auf 17x15 --
    beide in dieselbe Datei zu speichern wuerde sowohl einen ehrlichen
    Vergleich als auch den Ueberschreib-Schutz weiter unten unmoeglich
    machen (ein schwacher 17x15-Champion wuerde sonst nie einen starken
    9x9-Champion "schlagen" muessen, weil beide Zahlen nicht vergleichbar
    sind -- die Datei-Trennung macht dieses Problem von vornherein unmoeglich).
    """
    return os.path.join(MODEL_DIR, f"dqn_champion_{cols}x{rows}.pt")


def resolve_champion_path(cols: int, rows: int) -> str | None:
    """Pfad zu einer TATSAECHLICH VORHANDENEN Champion-Datei fuer dieses
    Brett -- oder None, wenn es (auf diesem Brett) noch keine gibt.

    Uebergangsregel: gibt es noch keine neue `dqn_champion_20x20.pt`, aber
    die alte `dqn_champion.pt` von vor der Brett-Umstellung, zaehlt die als
    20x20-Champion (Legacy-Lesepfad; geschrieben wird nur noch neu).
    """
    new_path = champion_path(cols, rows)
    if os.path.exists(new_path):
        return new_path
    if (cols, rows) == (20, 20) and os.path.exists(_LEGACY_CHAMPION_PATH):
        return _LEGACY_CHAMPION_PATH
    return None


def load_champion_config(path: str) -> DQNConfig | None:
    """Laedt die EXAKTEN Einstellungen, mit denen der unter `path` gespeicherte
    Champion trainiert wurde (siehe "full_config" in `_save_champion`).

    Ohne das wuerde jedes Weitertrainieren (CLI --weiter wie das Menue-Haekchen
    "Champion weitertrainieren") wieder mit den Code-Standardwerten aus
    config.py starten, obwohl der Champion vielleicht mit ganz anderen Werten
    (Netzgroesse, Lernrate, Fruechte, ...) gezuechtet wurde -- das Weiter-
    trainieren waere dann inkonsistent zu dem, was ihn tatsaechlich stark
    gemacht hat. Gibt None zurueck, wenn (noch) kein Champion existiert.
    `path` kommt typischerweise aus `resolve_champion_path(cols, rows)`.
    """
    if not os.path.exists(path):
        return None
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    full_config = checkpoint.get("full_config")
    if not full_config:
        return None
    # Nur bekannte Felder uebernehmen -- robust, falls DQNConfig sich seither
    # geaendert hat (neue Felder fehlen im alten Checkpoint, alte entfernte
    # Felder werden ignoriert statt einen TypeError zu werfen).
    valid_fields = {f.name for f in fields(DQNConfig)}
    kwargs = {k: v for k, v in full_config.items() if k in valid_fields}
    return DQNConfig(**kwargs)


@dataclass
class DQNStats:
    """Momentaufnahme des Trainings -- alles, was das Dashboard anzeigt."""
    # Fortschritt
    total_steps: int = 0          # Trainer-Ticks (ein Tick = alle Spiele je 1 Zug)
    total_moves: int = 0          # einzelne Schlangen-Zuege insgesamt
    total_episodes: int = 0       # abgeschlossene Partien insgesamt
    elapsed: float = 0.0          # Trainingszeit in Sekunden

    # Spielstaerke
    mean_score: float = 0.0       # gleitendes Mittel (mit Neugier -> pessimistisch)
    median_score: float = 0.0
    best_score: int = 0           # bester je erreichter Score (Gluecks-Partie)
    eval_score: float | None = None   # PRUEFUNG ohne Zufall = ehrlicher Wert
    eval_best: float = 0.0        # bester Pruefungs-Durchschnitt (= Champion)
    eval_max: int = 0             # bester Einzelwert in einer Pruefung
    mean_length: float = 0.0
    mean_steps: float = 0.0       # Ø Zuege pro Partie
    mean_steps_per_fruit: float | None = None   # Effizienz (kleiner = besser)
    fill_percent: float = 0.0     # wie viel vom Spielfeld die Schlange fuellt

    # Lernzustand
    epsilon: float = 1.0
    loss: float | None = None
    mean_q: float = 0.0
    buffer_fill: float = 0.0
    buffer_len: int = 0
    learn_steps: int = 0

    # Verlauf
    deaths: dict = field(default_factory=dict)
    recent_deaths: dict = field(default_factory=dict)   # nur die letzten 100
    curve: list = field(default_factory=list)           # Ø-Score im Training
    eval_curve: list = field(default_factory=list)      # Pruefungs-Score
    eval_points: list = field(default_factory=list)     # zugehoerige Episodennummern


class MultiGameTrainer:
    """Treibt `num_games` Spiele parallel an und trainiert EIN gemeinsames Netz."""

    def __init__(self, cfg: DQNConfig | None = None, seed: int | None = None,
                 log_to_csv: bool = True, resume_from: str | None = None) -> None:
        self.cfg = cfg or DQNConfig()
        cfg = self.cfg

        # Welche Wahrnehmung? Daraus folgt auch die Eingangsgroesse des Netzes.
        # cols/rows werden mitgegeben, weil "full_board" seine Eingangsgroesse
        # erst aus der Brettgroesse ableiten kann (siehe ai/perception.py).
        self.perceive, self.input_size, _labels = get_perception(
            cfg.perception, cfg.grid_cols, cfg.grid_rows)

        # Champion-Datei DIESES Bretts -- jede Brettgroesse hat ihre eigene
        # (siehe champion_path() oben im Modul).
        self.champion_file = champion_path(cfg.grid_cols, cfg.grid_rows)

        # Ein Basis-Seed sorgt dafuer, dass jedes Spiel seinen EIGENEN Zufall
        # bekommt (sonst spawnen alle Fruechte identisch -> die Spiele waeren
        # Klone und wuerden das Tagebuch mit lauter gleichen Erfahrungen fuellen).
        if seed is None:
            seed = cfg.seed
        base_seed = seed if seed is not None else random.randrange(1 << 30)
        self.base_seed = base_seed

        self.agent = DQNAgent(cfg, self.input_size, seed=base_seed)
        self.buffer = make_buffer(cfg, self.input_size,
                                  rng=np.random.default_rng(base_seed + 999))

        self.game_cfg = GameConfig(
            grid_cols=cfg.grid_cols,
            grid_rows=cfg.grid_rows,
            fruit_count=cfg.fruit_count,
            wrap_walls=cfg.wrap_walls,
        )
        self.games: list[SnakeGame] = [
            SnakeGame(self.game_cfg, rng=random.Random(base_seed + i))
            for i in range(cfg.num_games)
        ]
        # Eine n-Schritt-Kette pro Spiel: sie sammelt die letzten Zuege dieses
        # Spiels, bis eine vollstaendige Kette fertig ist (siehe memory.py).
        self.chains = [NStepChain(cfg.n_step, cfg.gamma) for _ in self.games]

        # Aktueller Wahrnehmungsvektor + Fruchtabstand pro Spiel.
        self.states = np.stack([self.perceive(g) for g in self.games])
        self.dists = [fruit_distance(g) for g in self.games]
        self.episode_steps = [0] * cfg.num_games

        # Optional: mit einem bereits trainierten Gehirn weitermachen.
        self.resumed_from: str | None = None
        if resume_from:
            self._resume(resume_from)

        # ---- Statistik ------------------------------------------------- #
        self.epsilon = cfg.eps_start
        self.total_steps = 0
        self.total_moves = 0
        self.total_episodes = 0
        self.best_score = 0
        self.started_at = time.time()
        self.recent_scores: deque[int] = deque(maxlen=100)
        self.recent_steps: deque[int] = deque(maxlen=100)
        self.recent_efficiency: deque[float] = deque(maxlen=100)
        self.recent_causes: deque[str] = deque(maxlen=100)
        self.deaths = {"wall": 0, "self": 0, "starvation": 0, "won": 0}
        self.loss_smoothed: float | None = None
        self.score_curve: list[float] = []
        self._curve_stride = 1

        # ---- Pruefung -------------------------------------------------- #
        self.eval_score: float | None = None
        self.eval_best = 0.0
        self.eval_max = 0
        self.eval_curve: list[float] = []
        self.eval_points: list[int] = []
        self._next_eval_at = cfg.eval_every_episodes
        self.champion_path: str | None = None
        # Bei einem Brett-Wechsel waehrend "--weiter" (siehe TRAININGSPLAN.md
        # S0.4): (altes_cols, altes_rows) wenn dieser Lauf ein Brett-Transfer
        # ist, sonst None. Nur fuers Dashboard/Log -- aendert kein Verhalten.
        self.board_transfer_from: tuple[int, int] | None = None
        # Protokoll der Meilenstein-Zeitplaene (1.3/2.4/2.5) -- damit im
        # Nachhinein exakt nachvollziehbar ist, WANN sich WAS geaendert hat.
        self.milestone_log: list[dict] = []
        # Rohdaten fuer den Post-Run-Report (ai/dqn/report.py): pro Episode
        # (Laenge, Ursache, Kopf-x, Kopf-y, Score, Zuege).
        self.episode_log: list[tuple] = []
        # Q-Kalibrierung (TRAININGSPLAN.md 0.1): (vorhergesagter Start-Q-Wert,
        # tatsaechlich erzielter Score) je Pruefpartie -- zeigt, ob das Netz
        # sich selbst uebermaessig ueberschaetzt.
        self.q_calibration_log: list[tuple[float, float]] = []
        # Loss-Verlauf, einmal pro Pruefung mitgeschrieben (nicht jeden Tick,
        # das waere zu viel) -- fuer die Report-Diagnose "Loss steigt".
        self.loss_history: list[float] = []

        # ---- Weitertrainieren: Zaehler + Neugier aus dem Checkpoint holen -- #
        # OHNE das hier wuerde jeder --weiter-Lauf so tun, als waere der Bot
        # frisch: Rekord auf 0, und -- viel schlimmer -- Neugier wieder auf
        # 100% Zufall (cfg.eps_start). Ein bereits gutes Netz wuerde dann erst
        # einmal minutenlang mit Zufallszuegen weitertrainiert und dabei aktiv
        # wieder verschlechtert, BEVOR die erste Pruefung ueberhaupt laeuft.
        # Stattdessen: Ticks/Episoden/Rekord uebernehmen und die Neugier aus den
        # uebernommenen Ticks neu berechnen (dieselbe Formel wie in step()) --
        # ist genug Erfahrung schon gesammelt, kommt dabei von selbst eine
        # niedrige Neugier heraus, statt dass wir sie erraten muessten.
        if self.resumed_from:
            meta = self._resume_meta
            self.total_steps = int(meta.get("total_steps", 0))
            self.total_episodes = int(meta.get("total_episodes", 0))

            # Brett-Transfer? (Champion kam von einem ANDEREN Brett, siehe
            # Klein-Feld-Curriculum in TRAININGSPLAN.md 2.3/S0.4). Erkennung:
            # die Brettmasse im Checkpoint weichen von der aktuellen Config ab.
            old_cols = int(meta.get("grid_cols", cfg.grid_cols))
            old_rows = int(meta.get("grid_rows", cfg.grid_rows))
            same_board = (old_cols == cfg.grid_cols and old_rows == cfg.grid_rows)

            if same_board:
                self.best_score = int(meta.get("best_train_score", meta.get("score", 0)))
                self.eval_best = float(meta.get("eval_mean", meta.get("score", 0.0)))
                self.eval_max = int(meta.get("score", 0))
            else:
                # Gewichte + gesammelte Erfahrung (s.u.) kommen mit, Rekorde
                # NICHT -- ein Score auf dem alten Brett ist mit dem neuen
                # nicht vergleichbar (anderes Feld, andere Schwierigkeit).
                self.best_score = 0
                self.eval_best = 0.0
                self.eval_max = 0
                self.board_transfer_from = (old_cols, old_rows)

            progress = min(1.0, self.total_steps / max(1, cfg.eps_decay_steps))
            self.epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * progress

        # ---- Bestehenden Champion NIE mit einem schlechteren ueberschreiben - #
        # Das gilt auch bei einem FRISCHEN Lauf (kein --weiter) UND bei einem
        # Brett-Transfer: ohne diese Pruefung startet eval_best dort bei 0, und
        # die allererste Pruefung mit eval_mean > 0 wuerde die Champion-Datei
        # DIESES Bretts sofort mit einem kaum trainierten Netz ueberschreiben --
        # ein bereits starker Champion (z.B. Pruefung 75) waere dann
        # unwiderruflich weg. Bewusst nur die Datei DIESES Bretts (siehe
        # resolve_champion_path) -- Scores anderer Brettgroessen sind nicht
        # vergleichbar. Der Datei-Wert ist die Untergrenze, egal ob dieser Lauf
        # neu startet oder weitertrainiert (bei --weiter auf DEMSELBEN Brett
        # ist es ohnehin derselbe Wert wie oben, also ein No-Op).
        _existing_path = resolve_champion_path(cfg.grid_cols, cfg.grid_rows)
        if _existing_path:
            try:
                import torch
                existing = torch.load(_existing_path, map_location="cpu", weights_only=False)
                existing_best = float(existing.get("eval_mean", existing.get("score", 0.0)))
                self.eval_best = max(self.eval_best, existing_best)
            except Exception:
                pass

        # Eigene Spiele fuer die Pruefung, damit das laufende Training nicht
        # gestoert wird (die Trainings-Partien laufen einfach weiter).
        # WICHTIG: hier auf cfg.eval_episodes NICHT deckeln (fruehere Version
        # kappte bei 16 -- eval_episodes=20 haette also still 4 Partien
        # "verloren", ohne dass es aufgefallen waere). Die Instanzen sind
        # leicht, ein hoeherer eval_episodes-Wert kostet praktisch nichts.
        self._eval_games = [
            SnakeGame(self.game_cfg, rng=random.Random(base_seed + 10_000 + i))
            for i in range(cfg.eval_episodes)
        ]

        # ---- Protokoll ------------------------------------------------- #
        self._csv_path: str | None = None
        self._run_id = time.strftime("%Y%m%d-%H%M%S")
        if log_to_csv:
            os.makedirs(LOG_DIR, exist_ok=True)
            self._csv_path = os.path.join(LOG_DIR, f"dqn-{self._run_id}.csv")
            with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([
                    "episode", "ticks", "sekunden", "epsilon", "o_score",
                    "median", "best", "pruefung", "o_laenge", "o_schritte",
                    "schritte_pro_frucht", "loss", "mean_q",
                    "tod_wand", "tod_selbst", "tod_hunger",
                ])
            self._write_run_info()

    # ------------------------------------------------------------------ #
    def _resume(self, path: str) -> None:
        """Laedt ein gespeichertes Gehirn als Startpunkt (Weitertrainieren)."""
        import torch
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if int(checkpoint.get("input_size", 11)) != self.input_size:
            raise ValueError(
                "Der gespeicherte Bot wurde mit einer anderen Wahrnehmung "
                f"trainiert ({checkpoint.get('perception')}) und passt nicht "
                f"zur aktuellen Einstellung ({self.cfg.perception})."
            )
        if tuple(checkpoint.get("hidden", ())) != tuple(self.cfg.hidden):
            raise ValueError("Der gespeicherte Bot hat eine andere Netzgroesse.")
        # Fehlt "activation" (Checkpoints von vor dieser Aenderung), war die
        # Aktivierung damals immer "tanh" -- das ist keine Vermutung, sondern
        # der einzige Wert, den es je gab, bevor dieses Feld eingefuehrt wurde.
        checkpoint_activation = checkpoint.get("activation", "tanh")
        if checkpoint_activation != getattr(self.cfg, "activation", "tanh"):
            raise ValueError(
                f"Der gespeicherte Bot nutzt Aktivierung '{checkpoint_activation}', "
                f"die aktuelle Einstellung ist '{self.cfg.activation}'."
            )
        self.agent.load_state_dict(checkpoint["state_dict"])
        self.resumed_from = path
        self._resume_meta = checkpoint

    # ================================================================== #
    # Ein Trainings-Tick: alle Spiele machen EINEN Zug, dann wird gelernt
    # ================================================================== #
    def step(self) -> None:
        cfg = self.cfg

        # 1) EIN Netz-Durchlauf fuer alle Spiele gleichzeitig (epsilon-greedy).
        actions = self.agent.act_batch(self.states, self.epsilon)

        # Einmal pro Tick berechnen (haengt nur von eval_best ab, nicht vom
        # einzelnen Spiel) -- siehe TRAININGSPLAN.md 2.4.
        formung_faktor = self.formung_faktor

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
            #    Ausserdem waere die Wahrnehmung nach einem Sieg gar nicht
            #    moeglich -- dann liegt keine Frucht mehr auf dem Feld.
            if terminal:
                next_state = self.states[i]
                dist_after = self.dists[i]
            else:
                next_state = self.perceive(game)
                dist_after = fruit_distance(game)

            # 4) Belohnung berechnen (reines Feedback, siehe reward.py).
            reward = compute_reward(cfg, result.ate_fruit, died,
                                    self.dists[i], dist_after,
                                    won=result.won, formung_faktor=formung_faktor)

            # 5) In die n-Schritt-Kette geben; fertige Ketten wandern ins
            #    gemeinsame Tagebuch. game.length wird mitgereicht, damit der
            #    Puffer spaeter gezielt einen Mindestanteil fortgeschrittener
            #    (langer) Erfahrungen ziehen kann (Laengen-Balance, siehe
            #    memory.py) -- ohne das bestuende das Tagebuch ueberwiegend
            #    aus Fruehspiel-Zuegen, weil jede Partie bei Laenge 3 beginnt.
            for item in self.chains[i].push(self.states[i], action_idx, reward,
                                            next_state, terminal, game.length):
                self.buffer.push(*item)

            # 6) Weiterschalten -- entweder naechste Situation oder neue Partie.
            if terminal:
                self._finish_episode(i, game, won=result.won)
            else:
                self.states[i] = next_state
                self.dists[i] = dist_after

        self.total_steps += 1
        self.total_moves += len(self.games)

        # 7) Aus dem Tagebuch lernen (macht nichts, solange zu wenig drin ist).
        if self.total_steps % cfg.train_every == 0:
            loss = self.agent.learn(self.buffer)
            if loss is not None:
                # Exponentielle Glaettung: der angezeigte Loss zappelt sonst so
                # stark, dass man den Trend nicht sieht.
                self.loss_smoothed = loss if self.loss_smoothed is None \
                    else 0.99 * self.loss_smoothed + 0.01 * loss

        # 8) Neugier linear abklingen lassen: viel ausprobieren -> vertrauen.
        #    eps_end_active statt cfg.eps_end: sinkt zusaetzlich einmalig ab
        #    einem Pruefungs-Meilenstein (TRAININGSPLAN.md 2.5) -- lange,
        #    gut gespielte Partien sollen nicht mehr an einem uebrig-
        #    gebliebenen Zufallszug sterben.
        progress = min(1.0, self.total_steps / max(1, cfg.eps_decay_steps))
        self.epsilon = cfg.eps_start + (self.eps_end_active - cfg.eps_start) * progress

        # 9) Faellige Pruefung? (kostet kurz Zeit, liefert den ehrlichen Wert)
        if self.total_episodes >= self._next_eval_at and len(self.buffer) >= cfg.min_buffer:
            self._next_eval_at = self.total_episodes + cfg.eval_every_episodes
            self.run_evaluation()
            self._apply_lr_milestone()

    # ------------------------------------------------------------------ #
    # Episode abschliessen (Statistik + Neustart dieses einen Spiels)
    # ------------------------------------------------------------------ #
    def _finish_episode(self, i: int, game: SnakeGame, won: bool) -> None:
        score = game.score
        steps = self.episode_steps[i]

        self.total_episodes += 1
        self.recent_scores.append(score)
        self.recent_steps.append(steps)
        if score > 0:
            self.recent_efficiency.append(steps / score)

        cause = "won" if won else (game.death_cause or "wall")
        self.deaths[cause] = self.deaths.get(cause, 0) + 1
        self.recent_causes.append(cause)

        # Fuer den Post-Run-Report (TRAININGSPLAN.md 0.1): Laenge, Ursache,
        # Todes-Position, Score, Zuege JEDER Episode -- das ist die Basis fuer
        # "Todesursachen nach Schlangenlaenge", die Kernauswertung des
        # Reports. game.head ist hier noch die Todesposition (reset() kommt
        # erst weiter unten).
        head_x, head_y = game.head
        self.episode_log.append((game.length, cause, head_x, head_y, score, steps))
        if len(self.episode_log) > 200_000:
            self.episode_log = self.episode_log[::2]   # wie score_curve: halbieren statt endlos wachsen

        if score > self.best_score:
            self.best_score = score

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
        self.chains[i].clear()
        self.states[i] = self.perceive(game)
        self.dists[i] = fruit_distance(game)
        self.episode_steps[i] = 0

    # ================================================================== #
    # PRUEFUNG: spielen ohne jeden Zufall -- das echte Koennen
    # ================================================================== #
    def run_evaluation(self) -> float:
        """Spielt `eval_episodes` Partien mit epsilon = 0 und mittelt die Scores.

        Laeuft auf EIGENEN Spielinstanzen, damit die laufenden Trainingspartien
        nicht unterbrochen werden. Es wird dabei nichts gelernt und nichts ins
        Tagebuch geschrieben -- das hier ist reine Messung, kein Training.
        """
        cfg = self.cfg
        # cfg.eval_episodes kann sich NACH dem Bau des Trainers geaendert
        # haben (z.B. train_dqn.py setzt fuer die Abschluss-Pruefung kurz
        # auf 30 hoch) -- ohne dieses Nachwachsen wuerde das wirkungslos
        # verpuffen, weil _eval_games sonst bei der Anzahl von __init__()
        # eingefroren bliebe.
        if len(self._eval_games) < cfg.eval_episodes:
            start = len(self._eval_games)
            self._eval_games.extend(
                SnakeGame(self.game_cfg, rng=random.Random(self.base_seed + 10_000 + start + i))
                for i in range(cfg.eval_episodes - start)
            )
        games = self._eval_games[:cfg.eval_episodes]
        for g in games:
            g.reset()

        states = np.stack([self.perceive(g) for g in games])
        alive = [True] * len(games)
        scores = [0] * len(games)
        steps = 0

        # Q-Kalibrierung (TRAININGSPLAN.md 0.1): was das Netz sich VOR der
        # Partie selbst zutraut (bester Q-Wert im Startzustand), im Report
        # spaeter gegen den TATSAECHLICH erzielten Score verglichen -- zeigt,
        # ob das Netz sich systematisch ueber- oder unterschaetzt.
        import torch
        with torch.no_grad():
            q_start = (
                self.agent.policy_net(
                    torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
                    .to(self.agent.device)
                )
                .max(dim=1).values.cpu().numpy()
            )

        # Wachsende Notbremse (TRAININGSPLAN.md 2.9): eval_max_steps=4000 ist
        # fuer einen mittelmaessigen Bot grosszuegig, wuerde aber ab einem
        # gewissen Niveau GUTE, lange Endspiel-Partien mitten im Vollmachen
        # abschneiden -- wir wuerden also unseren eigenen Fortschritt
        # kappen, ohne es zu merken. Die Grenze waechst deshalb mit dem
        # bisher besten Pruefungs-Durchschnitt mit (Formel: 50 Schritte pro
        # Punkt + 20 Punkte Puffer) und wird nie kleiner als der Config-Wert.
        max_steps = max(cfg.eval_max_steps, int(50 * (self.eval_best + 20)))

        while any(alive) and steps < max_steps:
            steps += 1
            idx_alive = [i for i, a in enumerate(alive) if a]
            actions = self.agent.act_batch(states[idx_alive], 0.0)
            for slot, i in enumerate(idx_alive):
                game = games[i]
                result = game.step_action(Action(int(actions[slot])))
                over = (not result.alive) or result.won
                if not over and game.steps_since_fruit >= cfg.starve_limit(game.length):
                    over = True
                if over:
                    alive[i] = False
                    scores[i] = game.score
                else:
                    states[i] = self.perceive(game)

        # Partien, die ins Zeitlimit gelaufen sind, mit aktuellem Stand werten.
        for i, a in enumerate(alive):
            if a:
                scores[i] = games[i].score

        self.q_calibration_log.extend(zip(q_start.tolist(), scores))
        if len(self.q_calibration_log) > 20_000:
            self.q_calibration_log = self.q_calibration_log[::2]
        if self.loss_smoothed is not None:
            self.loss_history.append(self.loss_smoothed)

        mean = sum(scores) / len(scores)
        self.eval_score = mean
        self.eval_max = max(self.eval_max, max(scores))
        self.eval_curve.append(mean)
        self.eval_points.append(self.total_episodes)

        # Champion = bester PRUEFUNGS-Durchschnitt (nicht die Gluecks-Partie).
        if mean > self.eval_best:
            self.eval_best = mean
            self._save_champion(mean, max(scores))
        return mean

    # ================================================================== #
    # Meilenstein-Zeitplaene (TRAININGSPLAN.md 1.3/2.4/2.5)
    # ================================================================== #
    # WICHTIG, Unterschied zu einem Auto-Tuner: Diese drei Mechanismen
    # reagieren auf ein ERREICHTES Pruefungs-NIVEAU (self.eval_best), NICHT
    # auf "hat sich seit X Episoden nicht verbessert". Sie sind deshalb
    # deterministisch und reproduzierbar -- zwei Laeufe mit identischer
    # Config nehmen IMMER denselben Weg, unabhaengig von Zufall/Rauschen in
    # der Pruefung. Genau das war Lucas Bedingung: "smart", aber weiterhin
    # exakt zurechenbar, welche Aenderung welche Wirkung hatte.
    def _milestone_scale(self) -> float:
        """Skaliert die (fuer 17x15 = 255 Zellen kalibrierten) Schwellen
        proportional zur tatsaechlichen Feldflaeche -- sonst waeren sie auf
        einem 9x9-Curriculum-Brett (max. 78 Punkte) nie erreichbar."""
        return (self.cfg.grid_cols * self.cfg.grid_rows) / 255.0

    @property
    def formung_faktor(self) -> float:
        """1.0 = volle Naeher/Weiter-Formung, 0.0 = komplett ausgeblendet.
        Faellt linear zwischen formung_aus_ab und formung_null_ab (siehe
        ai/dqn/reward.py fuer die Begruendung)."""
        cfg = self.cfg
        scale = self._milestone_scale()
        aus_ab = cfg.formung_aus_ab * scale
        null_ab = cfg.formung_null_ab * scale
        if null_ab <= aus_ab:
            return 0.0 if self.eval_best >= aus_ab else 1.0
        frac = (null_ab - self.eval_best) / (null_ab - aus_ab)
        return float(min(1.0, max(0.0, frac)))

    @property
    def eps_end_active(self) -> float:
        """Der gerade geltende Neugier-Boden -- sinkt einmalig auf
        eps_end_spaet, sobald eval_best die (brett-skalierte) Schwelle
        eps_spaet_ab erreicht hat."""
        cfg = self.cfg
        if self.eval_best >= cfg.eps_spaet_ab * self._milestone_scale():
            return cfg.eps_end_spaet
        return cfg.eps_end

    def _apply_lr_milestone(self) -> None:
        """Nach jeder Pruefung: waehlt die hoechste bereits erreichte Stufe
        aus cfg.lr_meilensteine und setzt die Optimizer-Lernrate darauf --
        idempotent (kein Effekt, wenn schon der richtige Wert gilt), deshalb
        ohne extra "schon gemacht"-Merker."""
        cfg = self.cfg
        scale = self._milestone_scale()
        target_lr = cfg.learning_rate
        for threshold, lr in sorted(cfg.lr_meilensteine):
            if self.eval_best >= threshold * scale:
                target_lr = lr
        current_lr = self.agent.optimizer.param_groups[0]["lr"]
        if abs(current_lr - target_lr) > 1e-12:
            for group in self.agent.optimizer.param_groups:
                group["lr"] = target_lr
            self.milestone_log.append({
                "episode": self.total_episodes,
                "eval_best": self.eval_best,
                "aenderung": f"Lernrate -> {target_lr}",
            })

    # ------------------------------------------------------------------ #
    # Speichern & Protokoll
    # ------------------------------------------------------------------ #
    def _save_champion(self, eval_mean: float, eval_max: int) -> None:
        cfg = self.cfg
        self.champion_path = self.agent.save_checkpoint(
            self.champion_file,
            {
                "score": eval_max,
                "eval_mean": eval_mean,
                "eval_episodes": cfg.eval_episodes,
                "best_train_score": self.best_score,
                "total_episodes": self.total_episodes,
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
                "grid_cols": cfg.grid_cols,
                "grid_rows": cfg.grid_rows,
                "fruit_count": cfg.fruit_count,
                "wrap_walls": cfg.wrap_walls,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                # Die kompletten Einstellungen dieses Laufs -- damit man beim
                # naechsten "--weiter" (oder einfach beim Nachschauen) sieht,
                # womit dieser Champion trainiert wurde, ohne im CSV-Log
                # danach suchen zu muessen.
                "full_config": {k: v for k, v in vars(cfg).items()},
            },
        )

    def _write_run_info(self) -> None:
        """Legt neben dem CSV eine JSON-Datei mit ALLEN Einstellungen des Laufs ab.

        Damit ist spaeter nachvollziehbar, welche Kurve zu welchen Parametern
        gehoert -- ohne das ist ein Vergleich zwischen zwei Laeufen wertlos.
        """
        info = {k: v for k, v in vars(self.cfg).items()}
        info["input_size"] = self.input_size
        info["seed"] = self.base_seed
        info["gestartet"] = time.strftime("%Y-%m-%d %H:%M:%S")
        info["resumed_from"] = self.resumed_from
        path = os.path.join(LOG_DIR, f"dqn-{self._run_id}-config.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, default=str, ensure_ascii=False)

    def write_report(self) -> tuple[str, str]:
        """Schreibt den Post-Run-Report (TRAININGSPLAN.md 0.1) neben CSV/JSON
        desselben Laufs. Aufrufstellen: run_headless (finally, auch bei
        Strg+C) und das Dashboard (Esc->Menue, Fenster schliessen)."""
        from ai.dqn.report import write_report
        os.makedirs(LOG_DIR, exist_ok=True)
        path_base = os.path.join(LOG_DIR, f"dqn-{self._run_id}")
        return write_report(self, path_base)

    def _log_row(self) -> None:
        """Schreibt alle 25 Episoden eine Zeile ins CSV-Protokoll."""
        if not self._csv_path or self.total_episodes % 25 != 0:
            return
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                self.total_episodes,
                self.total_steps,
                f"{time.time() - self.started_at:.1f}",
                f"{self.epsilon:.4f}",
                f"{self.mean_score():.3f}",
                f"{self.median_score():.1f}",
                self.best_score,
                "" if self.eval_score is None else f"{self.eval_score:.2f}",
                f"{self.mean_length():.2f}",
                f"{self.mean_episode_steps():.1f}",
                f"{self.efficiency():.2f}" if self.efficiency() else "",
                "" if self.loss_smoothed is None else f"{self.loss_smoothed:.5f}",
                f"{self.agent.last_mean_q:.3f}",
                self.deaths.get("wall", 0),
                self.deaths.get("self", 0),
                self.deaths.get("starvation", 0),
            ])

    # ------------------------------------------------------------------ #
    # Auswertung
    # ------------------------------------------------------------------ #
    def mean_score(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    def median_score(self) -> float:
        """Der typische Lauf -- unempfindlich gegen einzelne Ausreisser.

        Liegt der Median deutlich unter dem Durchschnitt, lebt die KI von
        wenigen Gluecks-Partien und ist noch nicht verlaesslich.
        """
        if not self.recent_scores:
            return 0.0
        return float(np.median(np.fromiter(self.recent_scores, dtype=np.float64)))

    def mean_length(self) -> float:
        return self.mean_score() + 3.0   # Startlaenge 3 + eine Zelle pro Frucht

    def mean_episode_steps(self) -> float:
        if not self.recent_steps:
            return 0.0
        return sum(self.recent_steps) / len(self.recent_steps)

    def efficiency(self) -> float | None:
        if not self.recent_efficiency:
            return None
        return sum(self.recent_efficiency) / len(self.recent_efficiency)

    def stats(self) -> DQNStats:
        """Momentaufnahme fuer das Dashboard."""
        recent = {}
        for cause in self.recent_causes:
            recent[cause] = recent.get(cause, 0) + 1
        board = self.cfg.grid_cols * self.cfg.grid_rows
        reference = self.eval_score if self.eval_score is not None else self.mean_score()
        return DQNStats(
            total_steps=self.total_steps,
            total_moves=self.total_moves,
            total_episodes=self.total_episodes,
            elapsed=time.time() - self.started_at,
            mean_score=self.mean_score(),
            median_score=self.median_score(),
            best_score=self.best_score,
            eval_score=self.eval_score,
            eval_best=self.eval_best,
            eval_max=self.eval_max,
            mean_length=self.mean_length(),
            mean_steps=self.mean_episode_steps(),
            mean_steps_per_fruit=self.efficiency(),
            fill_percent=100.0 * (reference + 3.0) / board,
            epsilon=self.epsilon,
            loss=self.loss_smoothed,
            mean_q=self.agent.last_mean_q,
            buffer_fill=self.buffer.fill_ratio,
            buffer_len=len(self.buffer),
            learn_steps=self.agent.learn_steps,
            deaths=dict(self.deaths),
            recent_deaths=recent,
            curve=self.score_curve,
            eval_curve=self.eval_curve,
            eval_points=self.eval_points,
        )
