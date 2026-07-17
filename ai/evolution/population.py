"""Die Population -- eine Menge von N Netzen (Genomen), die zusammen "eine
Generation" bilden, und die Regel, wie daraus die naechste Generation entsteht.

Kernstueck ist `evolve(fitnesses)`: Aus den bewerteten Genomen dieser Generation
wird die naechste gezuechtet -- mit Elitismus, Selektion, Crossover, Mutation.
"""

from __future__ import annotations

import numpy as np

from ai.network import DEFAULT_HIDDEN, random_genome
from ai.evolution import genetics


class Population:
    """Haelt die Genome einer Generation und erzeugt die naechste Generation."""

    def __init__(
        self,
        size: int,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
        elitism: int = 2,
        mutation_rate: float = 0.05,
        mutation_strength: float = 0.2,
        tournament_k: int = 4,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.size = size
        self.hidden = hidden
        self.elitism = elitism                # so viele Beste kommen unveraendert weiter
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.tournament_k = tournament_k
        self.rng = rng or np.random.default_rng()

        # Generation 0: lauter zufaellige Netze (die crashen anfangs staendig).
        self.genomes: list[np.ndarray] = [
            random_genome(hidden, self.rng) for _ in range(size)
        ]
        self.generation = 0

    def evolve(self, fitnesses: np.ndarray) -> None:
        """Erzeugt aus den bewerteten Genomen die naechste Generation.

        Ablauf:
        1) ELITISMUS: Die besten `elitism` Genome werden UNVERAENDERT uebernommen.
           Das garantiert, dass die Population nie schlechter wird als ihr bisher
           bestes Individuum -- egal wie viel Mutations-Chaos sonst passiert.
        2) REST auffuellen: Wiederhole bis voll:
             a) zwei Eltern per Turnier-Selektion waehlen
             b) per Crossover ein Kind mischen
             c) das Kind mutieren
        """
        order = np.argsort(fitnesses)[::-1]  # Indizes von best -> schlecht
        new_genomes: list[np.ndarray] = []

        # 1) Elite unveraendert uebernehmen.
        for i in range(min(self.elitism, self.size)):
            new_genomes.append(self.genomes[order[i]].copy())

        # 2) Rest durch Zucht auffuellen.
        while len(new_genomes) < self.size:
            parent_a = genetics.tournament_select(self.genomes, fitnesses, self.tournament_k, self.rng)
            parent_b = genetics.tournament_select(self.genomes, fitnesses, self.tournament_k, self.rng)
            child = genetics.uniform_crossover(parent_a, parent_b, self.rng)
            child = genetics.mutate(child, self.mutation_rate, self.mutation_strength, self.rng)
            new_genomes.append(child)

        self.genomes = new_genomes
        self.generation += 1

    def diversity(self) -> float:
        """Mass fuer die Vielfalt der Population (mittlere Streuung ueber alle Gene).

        Sinkt dieser Wert stark, sind sich alle Netze sehr aehnlich geworden ->
        die Population "verrennt" sich womoeglich in einer Loesung und probiert
        nichts Neues mehr. Genau darauf will der User im Dashboard achten koennen.
        """
        matrix = np.stack(self.genomes)          # (N, genome_size)
        return float(np.mean(np.std(matrix, axis=0)))
