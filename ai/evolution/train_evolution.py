"""Der Neuroevolution-Trainer -- die Zucht-Maschine (ohne pygame, damit turbo-schnell).

Verantwortlich fuer:
- FITNESS: wie gut war ein Netz? (nur aus Punkten + Ueberlebenszeit -- KEINE
  versteckte Strategie-Belohnung, siehe Leitplanke in CLAUDE.md)
- VERHUNGERN-TIMEOUT: bricht Partien ab, in denen die Schlange ewig im Kreis
  faehrt, ohne zu fressen. Das ist eine TRAININGS-Regel (kein Spielgesetz) und
  der wichtigste Schutz gegen das "Verrennen" in die Kreisdreh-Sackgasse.
- GENERATIONEN-SCHLEIFE: in zwei Varianten:
    * Schritt-fuer-Schritt (begin/step/end_generation) -> das Dashboard kann
      zwischen den Zuegen zeichnen und man schaut live zu.
    * Turbo (run_generation_fast) -> alles ohne Verzoegerung, maximale Geschwindigkeit.
- STATISTIK + LOG: pro Generation detaillierte Kennzahlen (Fitness, Score,
  Todesursachen, Diversitaet) -- live im Dashboard und in eine CSV-Datei.
"""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from game.config import GameConfig
from game.snake_game import Action, SnakeGame
from ai.perception import perceive
from ai.network import DEFAULT_HIDDEN, NumpyPolicy, genome_to_net
from ai.evolution.population import Population

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models")


# --------------------------------------------------------------------------- #
# Konfiguration -- ALLE Stellschrauben an einem Ort (auch vom Dashboard nutzbar)
# --------------------------------------------------------------------------- #
@dataclass
class EvolutionConfig:
    # Population & Genetik
    population_size: int = 100
    hidden: tuple[int, ...] = DEFAULT_HIDDEN
    elitism: int = 4                 # so viele Beste bleiben unveraendert erhalten
    mutation_rate: float = 0.05      # Anteil veraenderter Gene
    mutation_strength: float = 0.2   # Staerke der Mutation
    tournament_k: int = 4            # Auswahldruck bei der Selektion

    # Spielumgebung fuers Training
    grid_cols: int = 20
    grid_rows: int = 20
    fruit_count: int = 1
    wrap_walls: bool = False

    # Trainings-Regeln (KEINE Spielregeln!)
    max_steps_without_fruit: int = 120   # Verhungern-Timeout
    max_steps: int = 1500                # harte Obergrenze pro Partie

    # Fitness-Gewichte: fitness = w_steps*Schritte + w_fruit*Fruechte + w_fruit_sq*Fruechte^2
    w_steps: float = 1.0
    w_fruit: float = 100.0
    w_fruit_sq: float = 120.0

    # Sonstiges
    seed: int | None = None


# --------------------------------------------------------------------------- #
# Ergebnis einer einzelnen Partie und Statistik einer Generation
# --------------------------------------------------------------------------- #
@dataclass
class GameResult:
    fitness: float
    score: int
    length: int
    steps: int
    cause: str  # "wall" | "self" | "starvation" | "timeout" | "won"


@dataclass
class GenerationStats:
    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    best_score: int
    mean_score: float
    mean_length: float
    mean_steps: float
    diversity: float
    deaths: dict = field(default_factory=dict)  # {wall, self, starvation, timeout, won}
    alltime_best_score: int = 0


def compute_fitness(cfg: EvolutionConfig, steps: int, fruits: int) -> float:
    """Fitness NUR aus Ueberlebenszeit + Punkten (keine Strategie-Hinweise)."""
    return cfg.w_steps * steps + cfg.w_fruit * fruits + cfg.w_fruit_sq * (fruits ** 2)


# --------------------------------------------------------------------------- #
# Der Trainer
# --------------------------------------------------------------------------- #
class EvolutionTrainer:
    """Haelt die Population, wertet Generationen aus und zuechtet die naechste."""

    def __init__(self, cfg: EvolutionConfig, log_to_csv: bool = True) -> None:
        self.cfg = cfg
        self.rng_np = np.random.default_rng(cfg.seed)
        self._seed_counter = int(cfg.seed) if cfg.seed is not None else random.randrange(1 << 30)

        self.population = Population(
            size=cfg.population_size,
            hidden=cfg.hidden,
            elitism=cfg.elitism,
            mutation_rate=cfg.mutation_rate,
            mutation_strength=cfg.mutation_strength,
            tournament_k=cfg.tournament_k,
            rng=self.rng_np,
        )

        self.history: list[GenerationStats] = []
        self.champion: dict | None = None  # {genome, score, fitness, generation}
        self.alltime_best_score = 0
        self.start_time = time.time()

        # Live-Zustand der aktuell laufenden Generation (fuer das Dashboard).
        self.games: list[SnakeGame] = []
        self.policies: list[NumpyPolicy] = []
        self.dones: list[bool] = []
        self.results: list[GameResult | None] = []
        self.generation_active = False

        # CSV-Log vorbereiten.
        self._csv_path = None
        if log_to_csv:
            os.makedirs(_LOG_DIR, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._csv_path = os.path.join(_LOG_DIR, f"evolution_{stamp}.csv")
            self._csv_header_written = False

    # ------------------------------------------------------------------ #
    # Schritt-fuer-Schritt-API (fuer das Live-Dashboard)
    # ------------------------------------------------------------------ #
    def begin_generation(self) -> None:
        """Setzt fuer jedes Genom eine frische Partie auf (noch nicht gespielt)."""
        self.games = []
        self.policies = []
        self.dones = []
        self.results = [None] * self.cfg.population_size

        for genome in self.population.genomes:
            game = SnakeGame(self._game_config(), rng=random.Random(self._next_seed()))
            self.games.append(game)
            self.policies.append(NumpyPolicy(genome, self.cfg.hidden))
            self.dones.append(False)

        self.generation_active = True

    def step_generation(self) -> bool:
        """Bewegt jede noch laufende Partie um GENAU einen Zug weiter.

        Gibt True zurueck, wenn die ganze Generation fertig ist (alle Partien tot
        oder abgebrochen). Das Dashboard ruft dies im Takt der Anzeige-Geschwindig-
        keit auf und zeichnet dazwischen.
        """
        if not self.generation_active:
            return True

        cfg = self.cfg
        all_done = True
        for i, game in enumerate(self.games):
            if self.dones[i]:
                continue

            obs = perceive(game)
            action = self.policies[i].act(obs)
            result = game.step_action(Action(action))

            ended = False
            if not result.alive:
                ended = True  # death_cause steht im Spiel (wall/self)
            elif result.won:
                game.death_cause = "won"
                ended = True
            elif game.steps_since_fruit >= cfg.max_steps_without_fruit:
                game.death_cause = "starvation"
                ended = True
            elif game.steps >= cfg.max_steps:
                game.death_cause = "timeout"
                ended = True

            if ended:
                self.dones[i] = True
                self.results[i] = self._make_result(game)
            else:
                all_done = False

        return all_done

    def end_generation(self) -> GenerationStats:
        """Wertet die fertige Generation aus und zuechtet die naechste."""
        fitnesses = np.array([r.fitness for r in self.results], dtype=np.float64)
        stats = self._build_stats(fitnesses)

        # Champion aktualisieren (bestes je gesehenes Netz nach Score, dann Fitness).
        best_idx = int(np.argmax(fitnesses))
        best_result = self.results[best_idx]
        if (self.champion is None
                or best_result.score > self.champion["score"]
                or (best_result.score == self.champion["score"]
                    and best_result.fitness > self.champion["fitness"])):
            self.champion = {
                "genome": self.population.genomes[best_idx].copy(),
                "score": best_result.score,
                "fitness": float(best_result.fitness),
                "generation": self.population.generation,
            }
            self._save_champion()

        self.history.append(stats)
        self._log_csv(stats)

        # Naechste Generation zuechten.
        self.population.evolve(fitnesses)
        self.generation_active = False
        return stats

    # ------------------------------------------------------------------ #
    # Turbo-API (ohne Anzeige, volle Geschwindigkeit)
    # ------------------------------------------------------------------ #
    def run_generation_fast(self) -> GenerationStats:
        """Wertet eine komplette Generation ohne jede Verzoegerung aus."""
        self.begin_generation()
        while not self.step_generation():
            pass
        return self.end_generation()

    # ------------------------------------------------------------------ #
    # Interne Helfer
    # ------------------------------------------------------------------ #
    def _game_config(self) -> GameConfig:
        return GameConfig(
            grid_cols=self.cfg.grid_cols,
            grid_rows=self.cfg.grid_rows,
            fruit_count=self.cfg.fruit_count,
            wrap_walls=self.cfg.wrap_walls,
        )

    def _next_seed(self) -> int:
        self._seed_counter += 1
        return self._seed_counter

    def _make_result(self, game: SnakeGame) -> GameResult:
        return GameResult(
            fitness=compute_fitness(self.cfg, game.steps, game.score),
            score=game.score,
            length=game.length,
            steps=game.steps,
            cause=game.death_cause or "timeout",
        )

    def _build_stats(self, fitnesses: np.ndarray) -> GenerationStats:
        scores = np.array([r.score for r in self.results])
        lengths = np.array([r.length for r in self.results])
        steps = np.array([r.steps for r in self.results])

        deaths = {"wall": 0, "self": 0, "starvation": 0, "timeout": 0, "won": 0}
        for r in self.results:
            deaths[r.cause] = deaths.get(r.cause, 0) + 1

        self.alltime_best_score = max(self.alltime_best_score, int(scores.max()))

        return GenerationStats(
            generation=self.population.generation,
            best_fitness=float(fitnesses.max()),
            mean_fitness=float(fitnesses.mean()),
            median_fitness=float(np.median(fitnesses)),
            best_score=int(scores.max()),
            mean_score=float(scores.mean()),
            mean_length=float(lengths.mean()),
            mean_steps=float(steps.mean()),
            diversity=self.population.diversity(),
            deaths=deaths,
            alltime_best_score=self.alltime_best_score,
        )

    def _save_champion(self) -> None:
        """Speichert das bisher beste Netz im PyTorch-Format + als Genom."""
        if self.champion is None:
            return
        os.makedirs(_MODEL_DIR, exist_ok=True)
        net = genome_to_net(self.champion["genome"], self.cfg.hidden)
        torch.save(
            {
                "state_dict": net.state_dict(),
                "hidden": self.cfg.hidden,
                "score": self.champion["score"],
                "generation": self.champion["generation"],
            },
            os.path.join(_MODEL_DIR, "evo_champion.pt"),
        )
        np.save(os.path.join(_MODEL_DIR, "evo_champion_genome.npy"), self.champion["genome"])

    def _log_csv(self, stats: GenerationStats) -> None:
        if not self._csv_path:
            return
        row = {
            "generation": stats.generation,
            "best_fitness": round(stats.best_fitness, 2),
            "mean_fitness": round(stats.mean_fitness, 2),
            "best_score": stats.best_score,
            "mean_score": round(stats.mean_score, 3),
            "mean_length": round(stats.mean_length, 2),
            "mean_steps": round(stats.mean_steps, 1),
            "diversity": round(stats.diversity, 5),
            "deaths_wall": stats.deaths["wall"],
            "deaths_self": stats.deaths["self"],
            "deaths_starvation": stats.deaths["starvation"],
            "deaths_timeout": stats.deaths["timeout"],
            "alltime_best_score": stats.alltime_best_score,
        }
        write_header = not self._csv_header_written
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)


# --------------------------------------------------------------------------- #
# Headless-Lauf ohne Dashboard (fuer schnelle Tests / reines Rechnen)
# --------------------------------------------------------------------------- #
def main(generations: int = 30, cfg: EvolutionConfig | None = None) -> None:
    cfg = cfg or EvolutionConfig()
    trainer = EvolutionTrainer(cfg)
    print(f"Neuroevolution headless: Population={cfg.population_size}, "
          f"Genom={trainer.population.genomes[0].size} Gewichte\n")
    for _ in range(generations):
        stats = trainer.run_generation_fast()
        d = stats.deaths
        print(f"Gen {stats.generation:3d} | bestScore {stats.best_score:2d} "
              f"(alltime {stats.alltime_best_score:2d}) | meanScore {stats.mean_score:5.2f} | "
              f"bestFit {stats.best_fitness:8.1f} | meanFit {stats.mean_fitness:7.1f} | "
              f"div {stats.diversity:.3f} | "
              f"Tod: Wand {d['wall']:3d} selbst {d['self']:3d} hunger {d['starvation']:3d}")
    print(f"\nChampion: Score {trainer.champion['score']} in Gen {trainer.champion['generation']}")
    print(f"Log: {trainer._csv_path}")


if __name__ == "__main__":
    main()
