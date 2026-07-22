"""Der Replay-Puffer -- das gemeinsame "Tagebuch" aller Schlangen.

Was ist das?
------------
Jedes Mal, wenn eine Schlange einen Zug macht, entsteht eine Erfahrung:

    "In DIESER Situation (state) habe ich DIESE Aktion (action) gewaehlt,
     dafuer gab es DIESE Belohnung (reward), danach sah es SO aus (next_state),
     und die Partie war danach zu Ende (done) -- oder eben nicht."

Diese fuenf Angaben nennt man eine *Transition*. Statt sofort daraus zu lernen
und sie dann wegzuwerfen, schreiben wir sie in ein Tagebuch und lernen spaeter
immer wieder aus ZUFAELLIGEN Seiten daraus. Das nennt sich **Experience Replay**.

Warum ist das so wichtig? Zwei Gruende, beide mit Alltagsvergleich:

1) *Seltene Erlebnisse zaehlen mehrfach.* Eine Frucht zu fressen passiert am
   Anfang vielleicht einmal pro 200 Zuege. Wuerde man diese eine Erfahrung nur
   ein einziges Mal zum Lernen benutzen, waere sie sofort wieder vergessen.
   Im Tagebuch wird sie dagegen noch hunderte Male zufaellig gezogen -- wie
   Vokabelkarten, die man mehrfach wiederholt, statt sie einmal zu lesen.

2) *Aufeinanderfolgende Zuege sind sich zu aehnlich.* Wer nur aus den letzten
   30 Zuegen lernt, lernt aus 30 fast identischen Situationen -- als wuerde man
   Autofahren lernen, indem man 30x dieselbe Kurve fahrt. Zufaellig gemischte
   Tagebuchseiten liefern bunt gemischte Situationen, und genau das braucht ein
   neuronales Netz, um stabil zu lernen.

Und der Clou fuer unser Setup: ALLE Spiele schreiben in DASSELBE Tagebuch.
Schlange 3 faehrt in die Wand -> dieser Fehler landet im gemeinsamen Puffer und
verbessert das gemeinsame Gehirn, das auch die Schlangen 1, 2, 4 und 5 benutzen.
So lernen die Spiele voneinander, ohne je miteinander zu "reden".

Technisch: Ringpuffer
---------------------
Der Puffer hat eine feste Groesse (z.B. 100.000 Eintraege). Ist er voll, wird
die aelteste Erinnerung ueberschrieben -- wie ein Notizbuch mit fester Seitenzahl,
in dem man vorne wieder anfaengt, wenn man hinten ankommt. Gespeichert wird in
vorab angelegten NumPy-Arrays (nicht in einer Python-Liste von Objekten): das ist
deutlich schneller und speichersparender, und eine Stichprobe ist damit nur ein
einziger Array-Zugriff.
"""

from __future__ import annotations

import numpy as np

from ai.perception import INPUT_SIZE


class ReplayBuffer:
    """Ringpuffer fuer Transitionen (state, action, reward, next_state, done).

    Alle fuenf Bestandteile liegen in eigenen, vorab angelegten NumPy-Arrays
    gleicher Laenge -- Eintrag i der einen Spalte gehoert zu Eintrag i aller
    anderen (wie Zeilen einer Tabelle).
    """

    def __init__(self, capacity: int, state_size: int = INPUT_SIZE,
                 rng: np.random.Generator | None = None) -> None:
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.rng = rng or np.random.default_rng()

        # Die "Tabellenspalten". float32 reicht voellig und halbiert den
        # Speicher gegenueber float64 (bei 100k x 11 Zahlen durchaus relevant).
        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)    # 0/1/2
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)    # 0.0 oder 1.0

        self._index = 0   # wo die naechste Erinnerung hingeschrieben wird
        self._size = 0    # wie viele Eintraege bereits gueltig sind

    # ------------------------------------------------------------------ #
    # Schreiben
    # ------------------------------------------------------------------ #
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Legt EINE Erfahrung ins Tagebuch (ueberschreibt ggf. die aelteste)."""
        i = self._index
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = 1.0 if done else 0.0

        # Zeiger einen weiter -- am Ende springt er wieder auf 0 (der "Ring").
        self._index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------ #
    # Lesen
    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int):
        """Zieht `batch_size` zufaellige Erfahrungen (mit Zuruecklegen).

        Rueckgabe: (states, actions, rewards, next_states, dones) als NumPy-
        Arrays. Das Umwandeln in PyTorch-Tensoren passiert bewusst erst im
        Agenten (agent.py) -- dieser Puffer bleibt torch-frei und damit leicht
        testbar.
        """
        idx = self.rng.integers(0, self._size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        """Wie viele Erfahrungen aktuell gespeichert sind (max. capacity)."""
        return self._size

    @property
    def fill_ratio(self) -> float:
        """Fuellstand 0.0 - 1.0 (fuers Dashboard)."""
        return self._size / self.capacity if self.capacity else 0.0
