"""Der genetische Algorithmus -- die drei Grundoperationen der "Zucht".

Alles arbeitet auf Genomen = flache NumPy-Vektoren (die Netzgewichte, siehe
network.py). Diese Datei enthaelt bewusst KEINE Spiel- oder Netz-Logik, nur die
reine Genetik. Dadurch ist sie einzeln testbar und leicht zu verstehen.

Die drei Bausteine:
- SELEKTION: Wer darf sich fortpflanzen? (Turnier-Auswahl -- fair und einfach)
- CROSSOVER: Zwei Eltern -> ein Kind (Gene gemischt)
- MUTATION: Kleine zufaellige Aenderungen an einzelnen Genen (bringt Neues rein)
"""

from __future__ import annotations

import numpy as np


def tournament_select(
    genomes: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Turnier-Selektion: waehlt k zufaellige Individuen, das fitteste gewinnt.

    Warum Turnier statt "immer die Allerbesten"? Es laesst auch mal etwas
    schwaechere Gene durch -> haelt die Vielfalt hoch und verhindert, dass die
    Population zu frueh in einer einzigen (evtl. mittelmaessigen) Loesung
    festhaengt ("verfruehte Konvergenz"). k steuert den Auswahldruck:
    groesseres k = strenger (nur sehr Gute), kleineres k = mehr Vielfalt.

    Gibt eine KOPIE des Gewinner-Genoms zurueck.
    """
    n = len(genomes)
    contenders = rng.integers(0, n, size=k)
    best = contenders[np.argmax(fitnesses[contenders])]
    return genomes[best].copy()


def uniform_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform-Crossover: jedes Gen kommt mit 50/50 von Elternteil A oder B.

    Bildlich: das Kind erbt jeden einzelnen "Regler" von einem der beiden Eltern.
    So werden gute Teil-Loesungen beider Eltern neu kombiniert.
    """
    mask = rng.random(parent_a.shape) < 0.5
    return np.where(mask, parent_a, parent_b).astype(np.float32)


def mutate(
    genome: np.ndarray,
    rate: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutation: verstellt einzelne Gene leicht zufaellig (Gauss-Rauschen).

    - rate: Anteil der Gene, die veraendert werden (z.B. 0.05 = 5 %).
    - strength: wie stark (Standardabweichung des addierten Rauschens).

    Mutation ist die einzige Quelle fuer voellig NEUE Ideen im Genpool. Zu wenig
    -> die Population erstarrt; zu viel -> gute Loesungen werden staendig wieder
    zerstoert. Beides ist im Dashboard einstellbar, damit man die Balance findet.
    Gibt ein neues (mutiertes) Genom zurueck; das Original bleibt unveraendert.
    """
    child = genome.copy()
    mask = rng.random(genome.shape) < rate
    noise = rng.normal(0.0, strength, size=genome.shape).astype(np.float32)
    child[mask] += noise[mask]
    return child
