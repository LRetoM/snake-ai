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

    # WICHTIG gegen Bewertung durch reines Zufallsglueck: jedes Genom spielt pro
    # Generation nicht nur 1, sondern mehrere unabhaengige Partien mit jeweils
    # eigenem zufaelligem Fruchtlayout. Die Fitness ist der DURCHSCHNITT ueber
    # alle Partien. Ein Netz, das nur in einem einzigen (leichten) Fruchtlayout
    # zufaellig gut abschneidet, aber in anderen Layouts versagt, wird so NICHT
    # mehr hoch bewertet -- nur wer ueber verschiedene Situationen hinweg
    # KONSISTENT gut ist, gilt als gut. Das ist der Hebel gegen "hat ein festes
    # Muster auswendig gelernt" und fuer echte Generalisierung.
    episodes_per_genome: int = 3

    # Spielumgebung fuers Training
    grid_cols: int = 20
    grid_rows: int = 20
    fruit_count: int = 1
    wrap_walls: bool = False

    # Trainings-Regeln (KEINE Spielregeln!)
    max_steps_without_fruit: int = 120   # Verhungern-Timeout
    max_steps: int = 1500                # harte Obergrenze pro Partie

    # Fitness-Gewichte: fitness = w_steps*Schritte + w_fruit*Fruechte + w_fruit_sq*Fruechte^2
    #                            + Effizienz-Bonus (siehe unten)
    w_steps: float = 1.0
    w_fruit: float = 100.0
    w_fruit_sq: float = 120.0

    # Effizienz-Bonus: belohnt KURZE Wege zur Frucht direkt, nicht nur "ueberlebt
    # lange genug". Bei jeder gefressenen Frucht gibt es +w_efficiency/Schritte-
    # seit-letzter-Frucht dazu -- je schneller die Frucht erreicht wurde, desto
    # groesser der Bonus. Das ist ein ANDERER Hebel als der Verhungern-Timeout:
    # der Timeout verhindert nur ewiges Nichtstun, belohnt aber nicht kuerzere
    # Wege (2 Wege von 10 bzw. 60 Schritten geben ohne diesen Bonus die gleiche
    # Fitness). Per A/B-Test bestaetigt: nur ein strengerer Timeout aendert die
    # Schritte/Frucht-Effizienz NICHT, dieser direkte Bonus schon.
    w_efficiency: float = 40.0

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
    steps_per_fruit: float | None = None  # Effizienz dieser einen Partie


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
    mean_steps_per_fruit: float | None = None  # Effizienz-Kennzahl (niedriger = besser)
    generations_since_improvement: int = 0     # Stagnations-Zaehler
    best_avg_score: float = 0.0  # bestes GENOM, gemittelt ueber seine Episoden (robust)


def compute_fitness(cfg: EvolutionConfig, steps: int, fruits: int, efficiency_bonus: float = 0.0) -> float:
    """Fitness aus Ueberlebenszeit + Punkten + Effizienz-Bonus (keine Strategie-Hinweise:
    der Bonus belohnt nur "wie schnell wurde JEDE gefressene Frucht erreicht", nicht
    "in welche Richtung soll ich laufen" o.ae.)."""
    return (cfg.w_steps * steps + cfg.w_fruit * fruits + cfg.w_fruit_sq * (fruits ** 2)
            + efficiency_bonus)


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
        self._last_improvement_gen = 0   # fuer den Stagnations-Zaehler

        # Live-Zustand der aktuell laufenden Generation (fuer das Dashboard).
        self.games: list[SnakeGame] = []
        self.policies: list[NumpyPolicy] = []
        self.dones: list[bool] = []
        self.results: list[GameResult | None] = []
        self.efficiency_bonus: list[float] = []  # pro Individuum, waechst bei jeder Frucht
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
        """Setzt fuer jedes Genom MEHRERE frische Partien auf (noch nicht gespielt).

        Jedes Genom bekommt `episodes_per_genome` unabhaengige Partien mit je
        eigenem zufaelligem Fruchtlayout (siehe Kommentar bei
        EvolutionConfig.episodes_per_genome). self.games/policies/dones/results/
        efficiency_bonus sind dafuer FLACHE Listen der Laenge
        population_size * episodes_per_genome; Eintraege [g*K:(g+1)*K] gehoeren
        alle zu Genom g. end_generation() mittelt sie wieder pro Genom.
        """
        self.games = []
        self.policies = []
        self.dones = []
        k = self.cfg.episodes_per_genome
        n_total = self.cfg.population_size * k
        self.results = [None] * n_total
        self.efficiency_bonus = [0.0] * n_total
        # Nur noch laufende Partien werden pro Zug angefasst (siehe step_generation).
        # Schrumpft mit der Zeit -> spart Rechenzeit gerade dann, wenn nur noch
        # wenige "Nachzuegler" (typischerweise die guten, lang ueberlebenden
        # Genome) uebrig sind und der Rest der Population schon fertig ist.
        self._active_indices: list[int] = list(range(n_total))

        for genome in self.population.genomes:
            policy = NumpyPolicy(genome, self.cfg.hidden)  # 1x pro Genom, fuer alle K Episoden geteilt
            for _ in range(k):
                game = SnakeGame(self._game_config(), rng=random.Random(self._next_seed()))
                self.games.append(game)
                self.policies.append(policy)
                self.dones.append(False)

        self.generation_active = True

    def step_generation(self) -> bool:
        """Bewegt jede noch laufende Partie um GENAU einen Zug weiter.

        Gibt True zurueck, wenn die ganze Generation fertig ist (alle Partien tot
        oder abgebrochen). Das Dashboard ruft dies im Takt der Anzeige-Geschwindig-
        keit auf und zeichnet dazwischen.

        Performance: iteriert NUR ueber die noch aktiven Partien (self._active_
        indices), nicht ueber alle N jeden Zug mit einem "continue" fuer die schon
        fertigen. Bei grossen Populationen mit vielen frueh sterbenden Individuen
        (typisch in fruehen Generationen) und wenigen lang ueberlebenden Nachzueglern
        (typisch in spaeten Generationen) spart das viel wiederholtes Leerlaufen.
        """
        if not self.generation_active:
            return True

        cfg = self.cfg
        still_active = []
        for i in self._active_indices:
            game = self.games[i]

            obs = perceive(game)
            action = self.policies[i].act(obs)
            steps_before = game.steps_since_fruit  # fuer den Effizienz-Bonus (siehe unten)
            result = game.step_action(Action(action))

            if result.ate_fruit:
                # Je weniger Schritte seit der letzten Frucht, desto groesser der
                # Bonus -- das belohnt kurze Wege direkt, nicht nur "hat ueberlebt".
                time_taken = steps_before + 1
                self.efficiency_bonus[i] += cfg.w_efficiency / time_taken

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
                self.results[i] = self._make_result(game, self.efficiency_bonus[i])
            else:
                still_active.append(i)

        self._active_indices = still_active
        return not still_active

    def end_generation(self) -> GenerationStats:
        """Wertet die fertige Generation aus und zuechtet die naechste.

        WICHTIG: Selektion und Champion-Kuer laufen auf dem GENOM-GEMITTELTEN
        Ergebnis (Durchschnitt ueber episodes_per_genome Partien je Genom), nicht
        auf einer einzelnen Partie. Das ist der Schutz gegen Zufallsglueck: ein
        Genom, das nur in EINER von mehreren Partien gut abschneidet, aber sonst
        versagt, hat einen niedrigen Durchschnitt und setzt sich nicht durch --
        nur konsistent gute Genome werden Eltern der naechsten Generation.
        """
        k = self.cfg.episodes_per_genome
        n_pop = self.cfg.population_size

        genome_fitness = np.empty(n_pop, dtype=np.float64)
        genome_score = np.empty(n_pop, dtype=np.float64)
        for g in range(n_pop):
            chunk = self.results[g * k:(g + 1) * k]
            genome_fitness[g] = np.mean([r.fitness for r in chunk])
            genome_score[g] = np.mean([r.score for r in chunk])

        # Champion aktualisieren: bestes GENOM nach gemitteltem Score, dann Fitness.
        best_idx = int(np.argmax(genome_fitness))
        best_avg_score = float(genome_score[best_idx])
        best_avg_fitness = float(genome_fitness[best_idx])
        improved = (
            self.champion is None
            or best_avg_score > self.champion["score"]
            or (best_avg_score == self.champion["score"] and best_avg_fitness > self.champion["fitness"])
        )
        if improved:
            self.champion = {
                "genome": self.population.genomes[best_idx].copy(),
                "score": best_avg_score,
                "fitness": best_avg_fitness,
                "generation": self.population.generation,
            }
            self._save_champion()
            self._last_improvement_gen = self.population.generation

        # Stagnations-Zaehler bezieht sich bewusst auf den ROBUSTEN Champion-
        # Fortschritt (gemittelter Score), nicht auf eine einzelne Gluecks-Partie.
        stagnation = self.population.generation - self._last_improvement_gen

        stats = self._build_stats(genome_fitness, genome_score, stagnation)
        self.history.append(stats)
        self._log_csv(stats)

        # Naechste Generation zuechten -- Selektion nach gemittelter Fitness.
        self.population.evolve(genome_fitness)
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

    def _make_result(self, game: SnakeGame, efficiency_bonus: float) -> GameResult:
        return GameResult(
            fitness=compute_fitness(self.cfg, game.steps, game.score, efficiency_bonus),
            score=game.score,
            length=game.length,
            steps=game.steps,
            cause=game.death_cause or "timeout",
            steps_per_fruit=(game.steps / game.score) if game.score > 0 else None,
        )

    def _build_stats(
        self, genome_fitness: np.ndarray, genome_score: np.ndarray, stagnation: int
    ) -> GenerationStats:
        """Baut die Anzeige-Statistik. genome_fitness/genome_score sind bereits pro
        Genom ueber seine episodes_per_genome Partien gemittelt (robust); die
        Verteilungs-Kennzahlen (mean_score, Todesursachen, Effizienz) laufen
        dagegen ueber ALLE einzelnen Partien (self.results, mehr Stichproben)."""
        scores = np.array([r.score for r in self.results])
        lengths = np.array([r.length for r in self.results])
        steps = np.array([r.steps for r in self.results])

        deaths = {"wall": 0, "self": 0, "starvation": 0, "timeout": 0, "won": 0}
        for r in self.results:
            deaths[r.cause] = deaths.get(r.cause, 0) + 1

        # "alltime_best_score" bleibt bewusst die rohe Einzelpartie-Bestmarke
        # (zeigt die Ausnahme-Spitzenleistung); der ROBUSTE Fortschritt steckt
        # im Champion (siehe best_avg_score) und im Stagnations-Zaehler.
        self.alltime_best_score = max(self.alltime_best_score, int(scores.max()))

        # Effizienz: Schritte pro Frucht, gemittelt ueber alle Partien, die
        # ueberhaupt mindestens 1 Frucht gefressen haben (niedriger = effizienter).
        eff_values = [r.steps_per_fruit for r in self.results if r.steps_per_fruit is not None]
        mean_eff = float(np.mean(eff_values)) if eff_values else None

        return GenerationStats(
            generation=self.population.generation,
            best_fitness=float(genome_fitness.max()),
            mean_fitness=float(genome_fitness.mean()),
            median_fitness=float(np.median(genome_fitness)),
            best_score=int(scores.max()),
            mean_score=float(scores.mean()),
            mean_length=float(lengths.mean()),
            mean_steps=float(steps.mean()),
            diversity=self.population.diversity(),
            deaths=deaths,
            alltime_best_score=self.alltime_best_score,
            mean_steps_per_fruit=mean_eff,
            generations_since_improvement=stagnation,
            best_avg_score=float(genome_score.max()),
        )

    def _save_champion(self) -> None:
        """Speichert das bisher beste Netz im PyTorch-Format + als Genom.

        Wichtig: Die Umgebungs-Einstellungen (Fruchtanzahl, Wandmodus, Feldgroesse)
        werden MIT gespeichert. So kann eine spaetere "KI zuschauen"-Ansicht die
        Schlange fair unter genau den Bedingungen testen, unter denen sie
        trainiert wurde -- ein Champion, der mit 3 Fruechten gezuechtet wurde,
        soll nicht ploetzlich mit nur 1 Frucht bewertet werden.
        """
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
                "grid_cols": self.cfg.grid_cols,
                "grid_rows": self.cfg.grid_rows,
                "fruit_count": self.cfg.fruit_count,
                "wrap_walls": self.cfg.wrap_walls,
                "episodes_per_genome": self.cfg.episodes_per_genome,
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
            "best_avg_score": round(stats.best_avg_score, 3),
            "mean_steps_per_fruit": (
                round(stats.mean_steps_per_fruit, 2) if stats.mean_steps_per_fruit is not None else ""
            ),
            "generations_since_improvement": stats.generations_since_improvement,
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
        eff = f"{stats.mean_steps_per_fruit:5.1f}" if stats.mean_steps_per_fruit is not None else "  n/a"
        print(f"Gen {stats.generation:3d} | bestAvgScore {stats.best_avg_score:5.2f} "
              f"(einzeln {stats.best_score:2d}, alltime {stats.alltime_best_score:2d}) | "
              f"meanScore {stats.mean_score:5.2f} | Effizienz {eff} Schr/Frucht | "
              f"stagn {stats.generations_since_improvement:3d} Gen | div {stats.diversity:.3f} | "
              f"Tod: Wand {d['wall']:3d} selbst {d['self']:3d} hunger {d['starvation']:3d}")
    print(f"\nChampion: Ø-Score {trainer.champion['score']:.2f} in Gen {trainer.champion['generation']}")
    print(f"Log: {trainer._csv_path}")


if __name__ == "__main__":
    main()
