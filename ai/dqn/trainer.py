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


def curriculum_path(cols: int, rows: int) -> str:
    """Datei mit gespeicherten Endspiel-Stellungen DIESES Bretts
    (TRAININGSPLAN.md 2.2) -- eigene Datei pro Brettgroesse aus demselben
    Grund wie bei champion_path(): eine Laenge-40-Stellung auf 9x9 ist etwas
    anderes als auf 17x15."""
    return os.path.join(MODEL_DIR, f"startstellungen_{cols}x{rows}.pkl")


def runbest_path(cols: int, rows: int) -> str:
    """Bester Stand des AKTUELLEN Laufs (nicht des Bretts insgesamt).

    Warum eine zweite Datei neben dem Champion? Die Champion-Datei ist durch
    die eval_best-Untergrenze geschuetzt: liegt dort schon ein starker Bot
    (z.B. Pruefung 87), speichert ein frischer Lauf, der "nur" 63 erreicht,
    GAR NICHTS -- 45 Minuten Training waeren komplett weg, obwohl der neue
    Bot vielleicht mit einer ANDEREN Wahrnehmung trainiert und fuer den
    naechsten Vergleich gebraucht wird. Hier landet deshalb immer der beste
    Stand des laufenden Trainings; die Champion-Datei bleibt unangetastet
    die Bestenliste des Bretts. Wird bei jedem Lauf-Start ueberschrieben.
    """
    return os.path.join(MODEL_DIR, f"dqn_runbest_{cols}x{rows}.pt")


# Laengen, bei denen run_evaluation() von der besten Pruefpartie eine
# Stellung fuers Endspiel-Curriculum sichert (TRAININGSPLAN.md 2.2).
_CURRICULUM_LENGTHS = (40, 50, 60)

# Trap-Erkennung fuer Curriculum-Stellungen (Lucas Idee 2026-07-24): eine
# gespeicherte Stellung gilt erst dann als aussichtslos, wenn JEDER der 3
# moeglichen ersten Zuege (geradeaus/links/rechts) mindestens so oft
# versucht wurde -- vorher wird nichts entfernt, auch wenn es bisher schlecht
# aussieht. "Versucht, aber gestorben" zaehlt nur, wenn es innerhalb weniger
# Zuege passierte (siehe _QUICK_DEATH_SCHRITTE) -- ein Versuch, der 80 Zuege
# ueberlebt hat, war ein ernsthafter Versuch, auch wenn er am Ende starb.
_CURRICULUM_MIN_VERSUCHE_PRO_PFAD = 5
_CURRICULUM_QUICK_DEATH_SCHRITTE = 15
_CURRICULUM_TRAP_SCHWELLE = 0.9   # ab dieser Quote gilt ein Pfad als aussichtslos


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
    eval_best: float = 0.0        # Champion-Schutzschwelle der Datei dieses Bretts
    eval_best_run: float = 0.0    # bester Pruefungs-Durchschnitt DIESES Laufs
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

        # Endspiel-Curriculum (TRAININGSPLAN.md 2.2). WICHTIG (Lucas Korrektur
        # 2026-07-24): gespeicherte Stellungen von der Platte werden NUR
        # geladen, wenn dieser Lauf tatsaechlich DENSELBEN Bot fortsetzt
        # (--weiter). Sonst koennten Stellungen von einem VOELLIG ANDEREN
        # frueheren Lauf hier landen (andere Wahrnehmung, ein ganz anderer
        # Trainingsstand) -- das widerspricht dem Grundgedanken des
        # Curriculums: die Stellungen sollen die EIGENEN, gerade erreichten
        # Grenzen DIESES Bots widerspiegeln, nicht die eines fremden Laufs.
        # Sichtbarer Fehler ohne diesen Fix: ein frisches Netz bekam ab
        # Episode 1 sofort Laenge-40+-Stellungen eines fremden, viel
        # staerkeren Bots vorgesetzt -- sah im Fenster kaputt aus und war es
        # konzeptionell auch. Ein frischer Lauf startet deshalb IMMER leer
        # und baut sich seinen Vorrat ausschliesslich aus seinen EIGENEN
        # Pruefungen selbst auf (siehe run_evaluation).
        self.curriculum_file = curriculum_path(cfg.grid_cols, cfg.grid_rows)
        self.curriculum_snapshots: list[dict] = (
            self._load_curriculum_snapshots() if resume_from else [])
        self.curriculum_traps_removed = 0

        # Ein Basis-Seed sorgt dafuer, dass jedes Spiel seinen EIGENEN Zufall
        # bekommt (sonst spawnen alle Fruechte identisch -> die Spiele waeren
        # Klone und wuerden das Tagebuch mit lauter gleichen Erfahrungen fuellen).
        if seed is None:
            seed = cfg.seed
        base_seed = seed if seed is not None else random.randrange(1 << 30)
        self.base_seed = base_seed
        # Eigener Zufall NUR fuer die Curriculum-Auswahl (TRAININGSPLAN.md
        # 2.2) -- getrennt vom Spiel-Zufall der einzelnen Spiele, damit ein
        # anderer curriculum_anteil nicht nebenbei auch die Fruchtspawns
        # verschiebt.
        self._curriculum_rng = random.Random(base_seed + 555)

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
        # Merker je Spiel: hat DIESE Partie aus einer Curriculum-Stellung
        # gestartet? Wandert mit ins episode_log, damit der Report natuerlich
        # gestartete Partien und Curriculum-Starts getrennt auswerten kann --
        # sonst verzerren die absichtlich lang startenden Curriculum-Partien
        # jede Laengen-Statistik, ohne dass man es sieht.
        self._from_curriculum = [False] * cfg.num_games
        # Trap-Erkennung (Lucas Idee 2026-07-24): manche gespeicherten
        # Stellungen koennten in Wirklichkeit schon unrettbar sein (die
        # beste Pruefpartie war beim Einsammeln zwar noch am Leben, aber der
        # Kaefig war vielleicht schon zu). Wuerde so eine Stellung immer
        # wieder geladen, sterben ALLE Partien von dort aus sofort wieder --
        # verschwendete Trainingszeit, ohne dass es auffaellt. Deshalb je
        # Spiel-Slot merken: AUS welcher Stellung (Objekt-Referenz) kam der
        # aktuelle Curriculum-Start, und welche der 3 moeglichen ERSTEN Zuege
        # (geradeaus/links/rechts) wurde versucht -- siehe step()/
        # _finish_episode() fuer die Auswertung.
        self._curriculum_snapshot_ref: list[dict | None] = [None] * cfg.num_games
        self._curriculum_first_action: list[int | None] = [None] * cfg.num_games
        # Auch die allererste Belegung darf schon aus dem Curriculum kommen
        # (TRAININGSPLAN.md 2.2, "Initialbelegung") -- sonst waeren die ersten
        # num_games Partien nach jedem Neustart systematisch kurz.
        for _i, _game in enumerate(self.games):
            self._reset_or_curriculum(_i, _game)
        del _i, _game
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
        # ZWEI getrennte Bestwerte -- das ist wichtig und war vorher EIN Wert,
        # was die Messung verfaelscht hat:
        #   eval_best     = Champion-SCHUTZSCHWELLE der Datei dieses Bretts
        #                   (kann von einem frueheren Lauf mit voellig anderer
        #                   Wahrnehmung stammen!). Entscheidet NUR, ob die
        #                   Champion-Datei ueberschrieben werden darf.
        #   eval_best_run = bestes Pruefungs-Mittel DIESES Laufs. Steuert die
        #                   Meilenstein-Mechanismen (Formung, Neugier-Boden,
        #                   Lernraten-Treppe) und alle Diagnosen im Report.
        # Vorher speiste die geerbte Schutzschwelle die Meilensteine: ein
        # FRISCHER Lauf startete dann mit abgeschalteter Formung und
        # abgesenktem Neugier-Boden, als waere er schon so gut wie der alte
        # Champion -- und der Report meldete "Plateau", weil er den neuen Lauf
        # an einem fremden Bestwert mass.
        self.eval_best = 0.0
        self.eval_best_run = 0.0
        self.eval_max = 0
        self.eval_curve: list[float] = []
        self.eval_points: list[int] = []
        # Volle Pruefungs-Historie fuer den Report: je Pruefung ein Eintrag mit
        # min/median/max, Zuege/s seit der letzten Pruefung, Loss, Start-Q --
        # damit sieht man z.B. Geschwindigkeits-Einbrueche im Reportverlauf
        # (haette den memory.py-Bug sofort sichtbar gemacht) statt nur den
        # Schnitt ueber den ganzen Lauf.
        self.eval_history: list[dict] = []
        self.eval_last_min: int | None = None
        self.eval_last_median: float | None = None
        self.eval_last_max: int | None = None
        self._rate_moves = 0
        self._rate_time = time.time()
        self._next_eval_at = cfg.eval_every_episodes
        self.champion_path: str | None = None
        self.runbest_file = runbest_path(cfg.grid_cols, cfg.grid_rows)
        self.runbest_saved: str | None = None
        # Woher stammt die Champion-Schutzschwelle (Wert + Wahrnehmung der
        # vorhandenen Datei)? Nur fuer Log/Report -- erklaert, warum ein Lauf
        # evtl. keinen Champion speichern konnte.
        self.champion_floor_info: dict | None = None
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
                # Beim Weitertrainieren desselben Bots IST sein altes Niveau
                # auch das Lauf-Niveau -- die Meilensteine sollen nicht wieder
                # von vorne anfangen (der Bot ist ja wirklich schon so gut).
                self.eval_best_run = self.eval_best
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
        # WICHTIG: nur self.eval_best (die Schutzschwelle der DATEI) wird hier
        # angehoben -- self.eval_best_run bleibt unberuehrt, denn der Wert der
        # Datei sagt nichts ueber das Koennen DIESES Laufs aus (er kann sogar
        # von einer anderen Wahrnehmung stammen).
        _existing_path = resolve_champion_path(cfg.grid_cols, cfg.grid_rows)
        if _existing_path:
            try:
                import torch
                existing = torch.load(_existing_path, map_location="cpu", weights_only=False)
                existing_best = float(existing.get("eval_mean", existing.get("score", 0.0)))
                self.eval_best = max(self.eval_best, existing_best)
                self.champion_floor_info = {
                    "wert": existing_best,
                    "wahrnehmung": existing.get("perception"),
                    "datei": _existing_path,
                }
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
                    "median", "best", "pruefung", "pruefung_min",
                    "pruefung_median", "pruefung_max", "o_laenge", "o_schritte",
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

    # ------------------------------------------------------------------ #
    # Endspiel-Curriculum (TRAININGSPLAN.md 2.2)
    # ------------------------------------------------------------------ #
    def _load_curriculum_snapshots(self) -> list[dict]:
        """Gespeicherte Endspiel-Stellungen dieses Bretts von der Platte, oder
        eine leere Liste, wenn es noch keine gibt bzw. die Datei beschaedigt
        ist (dann einfach ohne Curriculum weitermachen statt abzustuerzen)."""
        if not os.path.exists(self.curriculum_file):
            return []
        import pickle
        try:
            with open(self.curriculum_file, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            return []

    def _save_curriculum_snapshots(self) -> None:
        """Schreibt die aktuelle (auf 200 gedeckelte) Sammlung auf die Platte.

        NUR wenn dieser Lauf zur Abstammungslinie der Champion-Datei gehoert
        (als --weiter gestartet ODER selbst schon einen Champion
        geschrieben): die Datei wird ja genau dann wieder GELADEN, wenn
        jemand den Champion weitertrainiert -- ein beliebiger frischer
        Experimentier-Lauf (z.B. vom Auto-Tuner) wuerde sonst den Vorrat
        des Champions mit seinen eigenen, dazu nicht passenden Stellungen
        ueberschreiben (dieselbe Cross-Run-Kontamination wie beim Laden,
        nur rueckwaerts). Im Speicher funktioniert das Curriculum fuer den
        laufenden Lauf unabhaengig davon ganz normal."""
        if not (self.resumed_from or self.champion_path):
            return
        import pickle
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(self.curriculum_file, "wb") as fh:
            pickle.dump(self.curriculum_snapshots, fh)

    def _reset_or_curriculum(self, i: int, game: SnakeGame) -> None:
        """Setzt ein TRAININGS-Spiel zurueck -- mit `curriculum_anteil`
        Wahrscheinlichkeit aus einer gespeicherten Endspiel-Stellung statt
        immer bei Laenge 3 (TRAININGSPLAN.md 2.2). Gilt bewusst NUR fuer
        Trainingspartien (Aufrufer: __init__ und _finish_episode) -- Pruefungen
        (run_evaluation) bleiben immer bei Laenge 3, damit der Massstab rein
        bleibt. Die KI bekommt dabei keinerlei Zusatzinfo, nur der Startpunkt
        der Partie aendert sich. `i` ist der Spiel-Slot -- darueber merkt sich
        `_from_curriculum[i]` die Startart fuer die Report-Trennung."""
        game.reset()
        self._from_curriculum[i] = False
        self._curriculum_snapshot_ref[i] = None
        self._curriculum_first_action[i] = None
        if self.cfg.curriculum_anteil > 0 and self.curriculum_snapshots:
            if self._curriculum_rng.random() < self.cfg.curriculum_anteil:
                snap = self._curriculum_rng.choice(self.curriculum_snapshots)
                game.load_snapshot(snap["snake"], snap["fruits"],
                                   snap["direction"], snap["steps_since_fruit"])
                self._from_curriculum[i] = True
                # Merken WELCHE Stellung (Objekt-Referenz, nicht Index -- der
                # Vorrat kann sich zwischenzeitlich aendern) -- fuer die
                # Trap-Erkennung in _finish_episode().
                self._curriculum_snapshot_ref[i] = snap

    def _update_curriculum_stats(self, snap: dict, first_action: int | None,
                                  cause: str, steps: int) -> None:
        """Bucht das Ergebnis einer Curriculum-Partie auf den ZUERST
        versuchten Zug (geradeaus/links/rechts) und entfernt die Stellung,
        falls sie sich als Trap erwiesen hat -- siehe die Konstanten oben
        fuer die genauen Schwellen. Reine Trainings-Buchhaltung, veraendert
        nichts an Wahrnehmung/Belohnung/Aktionswahl."""
        if first_action is None:
            return   # Partie endete noch vor dem ersten Zug -- kann nicht passieren, aber sicher ist sicher
        pfade = snap.setdefault("pfade", {
            0: {"versuche": 0, "quick_deaths": 0},
            1: {"versuche": 0, "quick_deaths": 0},
            2: {"versuche": 0, "quick_deaths": 0},
        })
        eintrag = pfade[first_action]
        eintrag["versuche"] += 1
        quick_death = cause in ("self", "wall") and steps <= _CURRICULUM_QUICK_DEATH_SCHRITTE
        if quick_death:
            eintrag["quick_deaths"] += 1

        # Trap nur bestaetigt, wenn ALLE 3 Pfade genug getestet wurden UND
        # JEDER davon fast immer schnell gestorben ist -- ein einzelner
        # schlechter Pfad reicht nicht, die Stellung koennte trotzdem ueber
        # einen ANDEREN ersten Zug loesbar sein.
        for a in (0, 1, 2):
            p = pfade[a]
            if p["versuche"] < _CURRICULUM_MIN_VERSUCHE_PRO_PFAD:
                return
            if p["quick_deaths"] / p["versuche"] < _CURRICULUM_TRAP_SCHWELLE:
                return

        for idx, s in enumerate(self.curriculum_snapshots):
            if s is snap:
                del self.curriculum_snapshots[idx]
                break
        self.curriculum_traps_removed += 1
        self._save_curriculum_snapshots()
        self.milestone_log.append({
            "episode": self.total_episodes,
            "eval_best": self.eval_best_run,
            "aenderung": (
                f"Curriculum-Stellung entfernt (Trap: alle 3 ersten Zuege "
                f"sterben fast immer schnell, Laenge {len(snap['snake'])})"),
        })

    # ================================================================== #
    # Ein Trainings-Tick: alle Spiele machen EINEN Zug, dann wird gelernt
    # ================================================================== #
    def step(self) -> None:
        cfg = self.cfg

        # 1) EIN Netz-Durchlauf fuer alle Spiele gleichzeitig (epsilon-greedy).
        actions = self.agent.act_batch(self.states, self.epsilon)

        # Einmal pro Tick berechnen (haengt nur vom Lauf-Niveau ab, nicht vom
        # einzelnen Spiel) -- siehe TRAININGSPLAN.md 2.4 bzw. Ueberlebens-
        # Fokus in config.py.
        formung_faktor = self.formung_faktor
        pfad_fokus_aktuell = self.pfad_fokus_aktuell

        for i, game in enumerate(self.games):
            action_idx = int(actions[i])

            # Trap-Erkennung: den ERSTEN Zug nach einem Curriculum-Start
            # merken (genau einmal pro Partie -- bleibt danach gesetzt, bis
            # _reset_or_curriculum ihn wieder auf None setzt). Damit laesst
            # sich spaeter auswerten, ob WIRKLICH alle 3 moeglichen ersten
            # Zuege ausprobiert wurden, bevor eine Stellung als aussichtslos
            # gilt (siehe _finish_episode).
            if self._from_curriculum[i] and self._curriculum_first_action[i] is None:
                self._curriculum_first_action[i] = action_idx

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
                                    won=result.won, formung_faktor=formung_faktor,
                                    pfad_fokus_aktuell=pfad_fokus_aktuell)

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

        # Trap-Erkennung (Lucas Idee 2026-07-24): war das eine Curriculum-
        # Partie, das Ergebnis dem ZUERST versuchten Zug zubuchen und danach
        # pruefen, ob die Stellung sich als aussichtslos erwiesen hat.
        if self._from_curriculum[i] and self._curriculum_snapshot_ref[i] is not None:
            self._update_curriculum_stats(
                self._curriculum_snapshot_ref[i],
                self._curriculum_first_action[i], cause, steps)

        # Fuer den Post-Run-Report (TRAININGSPLAN.md 0.1): Laenge, Ursache,
        # Todes-Position, Score, Zuege JEDER Episode -- das ist die Basis fuer
        # "Todesursachen nach Schlangenlaenge", die Kernauswertung des
        # Reports. game.head ist hier noch die Todesposition (reset() kommt
        # erst weiter unten).
        head_x, head_y = game.head
        self.episode_log.append((game.length, cause, head_x, head_y, score, steps,
                                 self._from_curriculum[i]))
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

        # Frisches Spiel fuer diesen Slot -- ggf. aus dem Endspiel-Curriculum
        # statt immer bei Laenge 3 (TRAININGSPLAN.md 2.2, siehe _reset_or_curriculum).
        self._reset_or_curriculum(i, game)
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

        # Endspiel-Curriculum (TRAININGSPLAN.md 2.2): waehrend der Pruefung
        # merken wir uns pro Partie die Stellung, sobald sie zum ERSTEN Mal
        # Laenge 40/50/60 erreicht -- nachher wird nur die Sammlung der
        # BESTEN Pruefpartie behalten (siehe unten), der Rest verworfen.
        curriculum_hits: list[dict[int, dict]] = [{} for _ in games]
        next_target = [0] * len(games)

        while any(alive) and steps < max_steps:
            steps += 1
            idx_alive = [i for i, a in enumerate(alive) if a]
            actions = self.agent.act_batch(states[idx_alive], 0.0)
            for slot, i in enumerate(idx_alive):
                game = games[i]
                result = game.step_action(Action(int(actions[slot])))

                while (next_target[i] < len(_CURRICULUM_LENGTHS)
                       and game.length >= _CURRICULUM_LENGTHS[next_target[i]]):
                    threshold = _CURRICULUM_LENGTHS[next_target[i]]
                    curriculum_hits[i][threshold] = {
                        "snake": list(game.snake),
                        "fruits": set(game.fruits),
                        "direction": game.direction,
                        "steps_since_fruit": game.steps_since_fruit,
                    }
                    next_target[i] += 1

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

        # Nur von der BESTEN Pruefpartie dieser Pruefung Stellungen behalten
        # (nicht von jeder -- sonst wuerde auch Glueck/Mittelmass einsickern).
        best_i = max(range(len(games)), key=lambda i: scores[i])
        if curriculum_hits[best_i]:
            self.curriculum_snapshots.extend(curriculum_hits[best_i].values())
            self.curriculum_snapshots = self.curriculum_snapshots[-200:]
            self._save_curriculum_snapshots()

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

        # Streuung der Pruefung mitschreiben (min/median/max): ein Mittel von
        # 60 kann "stabil 55-65" ODER "mal 20, mal 100" bedeuten -- fuer die
        # Bewertung ist das ein riesiger Unterschied.
        self.eval_last_min = int(min(scores))
        self.eval_last_median = float(np.median(scores))
        self.eval_last_max = int(max(scores))

        # Zuege/s seit der letzten Pruefung: zeigt Geschwindigkeits-Einbrueche
        # IM VERLAUF (der memory.py-Bug von 2026-07-24 waere hier sofort als
        # fallende Kurve sichtbar gewesen, statt nur als Bauchgefuehl).
        now = time.time()
        rate = (self.total_moves - self._rate_moves) / max(1e-9, now - self._rate_time)
        self._rate_moves, self._rate_time = self.total_moves, now

        self.eval_history.append({
            "episode": self.total_episodes,
            "mittel": round(mean, 2),
            "min": self.eval_last_min,
            "median": round(self.eval_last_median, 1),
            "max": self.eval_last_max,
            "zuege_pro_s": int(rate),
            "loss": None if self.loss_smoothed is None else round(self.loss_smoothed, 4),
            "start_q": round(float(q_start.mean()), 2),
            "epsilon": round(self.epsilon, 4),
        })

        # Bester Stand DIESES Laufs: steuert Meilensteine/Diagnosen und wird
        # immer gesichert (runbest_path) -- unabhaengig davon, ob die
        # Champion-Schutzschwelle der Datei uebertroffen wird.
        if mean > self.eval_best_run:
            self.eval_best_run = mean
            self.runbest_saved = self.agent.save_checkpoint(
                self.runbest_file, self._checkpoint_meta(mean, max(scores)))

        # Champion = bester PRUEFUNGS-Durchschnitt (nicht die Gluecks-Partie),
        # und nur, wenn er die Schutzschwelle der Datei uebertrifft.
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
        ai/dqn/reward.py fuer die Begruendung). Bewusst an eval_best_RUN
        gekoppelt, nicht an die Champion-Schutzschwelle der Datei: ein
        frischer Lauf soll mit VOLLER Formung starten, auch wenn auf der
        Platte schon ein starker Champion liegt."""
        cfg = self.cfg
        scale = self._milestone_scale()
        aus_ab = cfg.formung_aus_ab * scale
        null_ab = cfg.formung_null_ab * scale
        if null_ab <= aus_ab:
            return 0.0 if self.eval_best_run >= aus_ab else 1.0
        frac = (null_ab - self.eval_best_run) / (null_ab - aus_ab)
        return float(min(1.0, max(0.0, frac)))

    @property
    def pfad_fokus_aktuell(self) -> float:
        """Der Pfad-Fokus-Regler (DQNConfig.pfad_fokus, 0=Sammeln..1=
        Ueberleben), multipliziert mit dem Ausblend-Faktor: faellt linear
        auf 0, sobald das Lauf-Niveau zwischen pfad_fokus_aus_ab und
        pfad_fokus_null_ab steigt (gleiche Mechanik wie formung_faktor,
        brett-skaliert) -- "erst den sicheren Pfad finden, dann ihn
        schneller machen". cfg.pfad_fokus=0 (Standard) macht diese
        Eigenschaft immer 0, unabhaengig vom Lauf-Niveau."""
        cfg = self.cfg
        if not cfg.pfad_fokus:
            return 0.0
        scale = self._milestone_scale()
        aus_ab = cfg.pfad_fokus_aus_ab * scale
        null_ab = cfg.pfad_fokus_null_ab * scale
        if null_ab <= aus_ab:
            fade = 0.0 if self.eval_best_run >= aus_ab else 1.0
        else:
            frac = (null_ab - self.eval_best_run) / (null_ab - aus_ab)
            fade = float(min(1.0, max(0.0, frac)))
        return cfg.pfad_fokus * fade

    @property
    def eps_end_active(self) -> float:
        """Der gerade geltende Neugier-Boden -- sinkt einmalig auf
        eps_end_spaet, sobald eval_best_run (das Niveau DIESES Laufs, siehe
        formung_faktor) die brett-skalierte Schwelle eps_spaet_ab erreicht."""
        cfg = self.cfg
        if self.eval_best_run >= cfg.eps_spaet_ab * self._milestone_scale():
            return cfg.eps_end_spaet
        return cfg.eps_end

    def _apply_lr_milestone(self) -> None:
        """Nach jeder Pruefung: waehlt die hoechste bereits erreichte Stufe
        aus cfg.lr_meilensteine und setzt die Optimizer-Lernrate darauf --
        idempotent (kein Effekt, wenn schon der richtige Wert gilt), deshalb
        ohne extra "schon gemacht"-Merker. Nutzt eval_best_run (Niveau DIESES
        Laufs, siehe formung_faktor)."""
        cfg = self.cfg
        scale = self._milestone_scale()
        target_lr = cfg.learning_rate
        for threshold, lr in sorted(cfg.lr_meilensteine):
            if self.eval_best_run >= threshold * scale:
                target_lr = lr
        current_lr = self.agent.optimizer.param_groups[0]["lr"]
        if abs(current_lr - target_lr) > 1e-12:
            for group in self.agent.optimizer.param_groups:
                group["lr"] = target_lr
            self.milestone_log.append({
                "episode": self.total_episodes,
                "eval_best": self.eval_best_run,
                "aenderung": f"Lernrate -> {target_lr}",
            })

    # ------------------------------------------------------------------ #
    # Speichern & Protokoll
    # ------------------------------------------------------------------ #
    def _checkpoint_meta(self, eval_mean: float, eval_max: int) -> dict:
        """Metadaten fuer Champion- UND Runbest-Checkpoints -- identisches
        Format, damit --weiter/watch_ai beide gleich laden koennen."""
        cfg = self.cfg
        return {
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
        }

    def _save_champion(self, eval_mean: float, eval_max: int) -> None:
        self.champion_path = self.agent.save_checkpoint(
            self.champion_file, self._checkpoint_meta(eval_mean, eval_max))

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
                "" if self.eval_last_min is None else self.eval_last_min,
                "" if self.eval_last_median is None else f"{self.eval_last_median:.1f}",
                "" if self.eval_last_max is None else self.eval_last_max,
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
            eval_best_run=self.eval_best_run,
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
