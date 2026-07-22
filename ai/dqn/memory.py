"""Das Erfahrungs-Tagebuch: Ringpuffer, n-Schritt-Ketten und Priorisierung.

Was ist eine "Transition"?
--------------------------
Jedes Mal, wenn eine Schlange einen Zug macht, entsteht eine Erfahrung:

    "In DIESER Situation (state) habe ich DIESE Aktion (action) gewaehlt,
     dafuer gab es DIESE Belohnung (reward), danach sah es SO aus (next_state),
     und die Partie war danach zu Ende (done) -- oder eben nicht."

Statt sofort daraus zu lernen und sie wegzuwerfen, schreiben wir sie in ein
Tagebuch und lernen spaeter immer wieder aus ZUFAELLIGEN Seiten daraus. Das
nennt sich **Experience Replay** und ist aus zwei Gruenden entscheidend:

1) *Seltene Erlebnisse zaehlen mehrfach.* Eine Frucht zu fressen passiert am
   Anfang vielleicht einmal pro 200 Zuege -- im Tagebuch wird dieses Erlebnis
   noch hunderte Male gezogen, wie eine Vokabelkarte, die man wiederholt.
2) *Aufeinanderfolgende Zuege sind sich zu aehnlich.* Nur aus den letzten 30
   Zuegen zu lernen waere, als wuerde man Autofahren lernen, indem man 30x
   dieselbe Kurve faehrt. Gemischte Tagebuchseiten sind viel lehrreicher.

ALLE Spiele schreiben in DASSELBE Tagebuch -- so lernen sie voneinander.

Drei Bausteine in dieser Datei
------------------------------
- `NStepChain`   : macht aus Einzelzuegen 3-Zug-Ketten (schnelleres Lernen)
- `ReplayBuffer` : der normale Ringpuffer, jede Erinnerung gleich wahrscheinlich
- `PrioritizedReplayBuffer` : zieht ueberraschende Erinnerungen oefter

n-Schritt-Ketten -- warum?
--------------------------
Im Standard-DQN lernt jede Erfahrung nur von ihrem direkten Nachfolger. Die
Erkenntnis "hier war der Weg zur Frucht falsch" muss also Zug fuer Zug
rueckwaerts sickern -- ein Dominostein pro Lernschritt. Mit 3-Zug-Ketten
speichern wir stattdessen "was in den naechsten DREI Zuegen zusammen passiert
ist". Das Wissen wandert dreimal so schnell rueckwaerts. Preis: die Rechnung
wird leicht ungenauer, solange die KI noch schlecht spielt (die drei Zuege
stammen ja von einer Politik, die sich gerade aendert). Bei n=3 ist dieser
Preis winzig und der Gewinn gross -- deshalb ist n=3 in der Praxis Standard.

Priorisierung -- "besser lernen, was wichtig ist"
-------------------------------------------------
Nicht jede Erinnerung ist gleich lehrreich. Ein Zug mitten im leeren Feld, bei
dem die Schaetzung des Netzes ohnehin schon stimmte, bringt fast nichts. Ein Zug,
bei dem sich das Netz stark VERSCHAETZT hat (grosser "TD-Fehler"), ist dagegen
genau die Stelle, an der noch etwas zu lernen ist.

Prioritized Experience Replay zieht deshalb ueberraschende Erinnerungen oefter --
wie ein Schueler, der die Karteikarten haeufiger wiederholt, bei denen er sich
geirrt hat, statt die zu pauken, die er laengst kann.

Damit das nicht schummelt, gibt es eine Korrektur: Erinnerungen, die oefter
gezogen werden, zaehlen beim Lernen entsprechend WENIGER stark (Fachbegriff:
Importance Sampling). Sonst haette die KI am Ende ein verzerrtes Weltbild, in
dem Katastrophen haeufiger vorkommen, als sie es in Wirklichkeit tun.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# =========================================================================== #
# n-Schritt-Kette (eine pro Spiel)
# =========================================================================== #
class NStepChain:
    """Sammelt n aufeinanderfolgende Zuege und macht daraus EINE Erfahrung.

    Aus (s0,a0,r0), (s1,a1,r1), (s2,a2,r2) wird:
        state      = s0
        action     = a0
        reward     = r0 + gamma*r1 + gamma^2*r2      (die Kette zusammengefasst)
        next_state = die Situation NACH dem dritten Zug
        discount   = gamma^3                          (so weit ist es gesprungen)

    Der `discount` wird MITGESPEICHERT, weil Ketten am Partie-Ende kuerzer sind
    (dann bleiben z.B. nur 2 Zuege uebrig -> gamma^2). Ohne diese Zahl wuerde das
    Lernziel fuer solche Rest-Ketten falsch berechnet.
    """

    def __init__(self, n_step: int, gamma: float) -> None:
        self.n = max(1, int(n_step))
        self.gamma = gamma
        self._items: deque = deque()

    def push(self, state, action, reward, next_state, done) -> list:
        """Nimmt einen Zug auf. Gibt die fertigen Erfahrungen zurueck (0, 1 oder
        -- am Partie-Ende -- mehrere).

        Die beiden Zustaende werden KOPIERT. Das ist keine Vorsicht, sondern
        zwingend: der Trainer haelt alle Wahrnehmungen in EINER grossen Tabelle
        und reicht hier nur eine Zeile davon herein. Diese Zeile ist ein
        Verweis, kein eigener Wert -- im naechsten Zug wird genau diese Zeile
        ueberschrieben. Ohne Kopie wuerde sich die vor 3 Zuegen gemerkte
        Situation nachtraeglich in die AKTUELLE verwandeln, und die KI wuerde
        aus voellig falschen Paaren lernen (der Fehler kostete im Vergleichs-
        test rund 90% der Spielstaerke -- und faellt sonst nirgends auf).
        """
        self._items.append((np.array(state, dtype=np.float32),
                            action, reward,
                            np.array(next_state, dtype=np.float32), done))
        ready = []

        if done:
            # Partie vorbei: alle angefangenen Ketten abschliessen. Sie enden
            # alle in derselben Endsituation, nur mit unterschiedlicher Laenge.
            while self._items:
                ready.append(self._make(len(self._items)))
                self._items.popleft()
            return ready

        if len(self._items) >= self.n:
            ready.append(self._make(self.n))
            self._items.popleft()
        return ready

    def _make(self, length: int):
        """Fasst die ersten `length` gepufferten Zuege zu einer Erfahrung zusammen."""
        state, action = self._items[0][0], self._items[0][1]
        total = 0.0
        discount = 1.0
        for i in range(length):
            total += discount * self._items[i][2]
            discount *= self.gamma
        last = self._items[length - 1]
        return (state, action, total, last[3], last[4], discount)

    def clear(self) -> None:
        self._items.clear()


# =========================================================================== #
# Normaler Ringpuffer
# =========================================================================== #
class ReplayBuffer:
    """Ringpuffer fester Groesse: neue Erinnerung rein, aelteste raus.

    Wie ein Notizbuch mit fester Seitenzahl -- ist es voll, faengt man vorne
    wieder an. Dass es dauerhaft "100% gefuellt" anzeigt, ist also normal und
    gewollt: die KI lernt immer aus den letzten N Erfahrungen, und deren
    QUALITAET steigt, waehrend sie besser spielt.

    Alles liegt in vorab angelegten NumPy-Arrays (nicht in Python-Listen):
    schneller, speichersparender, und eine Stichprobe ist ein Array-Zugriff.
    """

    def __init__(self, capacity: int, state_size: int,
                 rng: np.random.Generator | None = None) -> None:
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.rng = rng or np.random.default_rng()

        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)

        self._index = 0
        self._size = 0

    # ---------------------------------------------------------------- #
    def push(self, state, action, reward, next_state, done, discount) -> None:
        """Legt EINE Erfahrung ins Tagebuch (ueberschreibt ggf. die aelteste)."""
        i = self._index
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = 1.0 if done else 0.0
        self.discounts[i] = discount
        self._on_push(i)
        self._index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _on_push(self, i: int) -> None:
        """Haken fuer die priorisierte Variante (hier absichtlich leer)."""

    # ---------------------------------------------------------------- #
    def sample(self, batch_size: int):
        """Zieht `batch_size` zufaellige Erfahrungen -- alle gleich wahrscheinlich.

        Rueckgabe: (states, actions, rewards, next_states, dones, discounts,
                    indices, weights). `weights` sind hier alle 1.0; die Felder
                    gibt es, damit der Agent beide Puffer-Arten gleich behandeln
                    kann.
        """
        idx = self.rng.integers(0, self._size, size=batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        return self._gather(idx, weights)

    def _gather(self, idx, weights):
        return (
            self.states[idx], self.actions[idx], self.rewards[idx],
            self.next_states[idx], self.dones[idx], self.discounts[idx],
            idx, weights,
        )

    def update_priorities(self, idx, td_errors) -> None:
        """Nur fuer die priorisierte Variante relevant -- hier passiert nichts."""

    def __len__(self) -> int:
        return self._size

    @property
    def fill_ratio(self) -> float:
        return self._size / self.capacity if self.capacity else 0.0


# =========================================================================== #
# Summenbaum -- das Werkzeug hinter der Priorisierung
# =========================================================================== #
class SumTree:
    """Erlaubt "ziehe zufaellig, aber gewichtet" in Blitzgeschwindigkeit.

    Naiv muesste man fuer eine gewichtete Ziehung alle 300.000 Gewichte
    aufsummieren -- viel zu langsam, wenn man das tausende Male pro Sekunde
    macht. Ein Summenbaum speichert stattdessen Zwischensummen: jeder Knoten
    kennt die Summe seiner beiden Kinder. Ziehen heisst dann nur noch, von oben
    ~19 Mal "links oder rechts?" zu entscheiden.

    Bild dazu: statt in einem Telefonbuch jede Seite einzeln durchzuzaehlen,
    schlaegt man in der Mitte auf und halbiert sich zum Ziel.

    Alle Operationen sind bewusst mit NumPy fuer den ganzen Stapel auf einmal
    geschrieben (nicht in einer Python-Schleife pro Eintrag) -- sonst waere die
    Priorisierung langsamer als der Lerngewinn, den sie bringt.
    """

    def __init__(self, capacity: int) -> None:
        size = 1
        while size < capacity:
            size *= 2
        self.size = size                                  # Anzahl Blaetter
        self.tree = np.zeros(2 * size, dtype=np.float64)  # 1 = Wurzel

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def set_one(self, i: int, value: float) -> None:
        """Schnellweg fuer EINEN Eintrag (wird bei jedem neuen Zug gebraucht).

        Der Stapel-Weg unten ist fuer 256 Eintraege auf einmal gebaut; fuer einen
        einzelnen waere sein NumPy-Aufwand groesser als die eigentliche Rechnung.
        Diese schlichte Schleife ueber ~19 Ebenen ist hier um ein Vielfaches
        schneller -- und sie laeuft bei jedem einzelnen Spielzug.
        """
        tree = self.tree
        pos = i + self.size
        tree[pos] = value
        pos //= 2
        while pos >= 1:
            tree[pos] = tree[2 * pos] + tree[2 * pos + 1]
            pos //= 2

    def set(self, idx: np.ndarray, values: np.ndarray) -> None:
        """Setzt die Gewichte an den Positionen `idx` und aktualisiert den Baum."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        values = np.atleast_1d(np.asarray(values, dtype=np.float64))
        # Doppelte Positionen zusammenfassen (beim Ziehen mit Zuruecklegen
        # kann derselbe Eintrag mehrfach vorkommen) -- sonst wuerde die
        # Summenrechnung im Baum durcheinandergeraten.
        idx, keep = np.unique(idx, return_index=True)
        pos = idx + self.size
        self.tree[pos] = values[keep]
        # Von unten nach oben alle betroffenen Summen neu bilden.
        pos = np.unique(pos // 2)
        while pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            if pos[0] == 1:
                break
            pos = np.unique(pos // 2)

    def sample(self, n: int, rng: np.random.Generator):
        """Zieht n Positionen, Wahrscheinlichkeit proportional zum Gewicht.

        Geschichtete Ziehung: der Gesamtbereich wird in n gleich grosse
        Abschnitte geteilt und aus jedem genau einmal gezogen. Das streut die
        Stichprobe gleichmaessiger als reines Wuerfeln (weniger Zufallsklumpen).
        """
        total = self.total
        if total <= 0.0:
            raise ValueError("SumTree ist leer")
        step = total / n
        targets = (np.arange(n) + rng.random(n)) * step

        pos = np.ones(n, dtype=np.int64)
        while pos[0] < self.size:
            left = 2 * pos
            left_sum = self.tree[left]
            go_right = targets > left_sum
            targets = targets - np.where(go_right, left_sum, 0.0)
            pos = left + go_right
        leaf_values = self.tree[pos]
        return pos - self.size, leaf_values


class PrioritizedReplayBuffer(ReplayBuffer):
    """Tagebuch, das ueberraschende Erinnerungen oefter zieht.

    - `alpha` steuert, WIE stark priorisiert wird (0 = gar nicht = normales
      Tagebuch, 1 = streng nach Fehlergroesse). 0.6 ist der uebliche Mittelweg.
    - `beta` steuert die Gegenkorrektur (Importance Sampling). Sie startet bei
      0.4 und waechst auf 1.0: am Anfang darf die KI ruhig etwas "verzerrt"
      lernen (Hauptsache schnell), gegen Ende soll das Weltbild wieder ehrlich
      sein.
    """

    def __init__(self, capacity: int, state_size: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_steps: int = 200_000, epsilon: float = 0.01,
                 rng: np.random.Generator | None = None) -> None:
        super().__init__(capacity, state_size, rng)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = max(1, int(beta_steps))
        self.epsilon = epsilon
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
        self._beta_progress = 0

    def _on_push(self, i: int) -> None:
        # Neue Erfahrungen bekommen die hoechste bisher gesehene Prioritaet --
        # so wird garantiert jede neue Erinnerung mindestens einmal gezogen
        # (man weiss ja noch nicht, wie ueberraschend sie ist).
        self.tree.set_one(i, self.max_priority)

    @property
    def beta(self) -> float:
        t = min(1.0, self._beta_progress / self.beta_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def sample(self, batch_size: int):
        idx, priorities = self.tree.sample(batch_size, self.rng)
        idx = np.clip(idx, 0, max(0, self._size - 1))

        # Gegenkorrektur: haeufiger gezogen -> beim Lernen schwaecher gewichtet.
        probs = priorities / self.tree.total
        probs = np.maximum(probs, 1e-12)
        weights = (self._size * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        self._beta_progress += 1
        return self._gather(idx, weights)

    def update_priorities(self, idx, td_errors) -> None:
        """Nach dem Lernen: neue Ueberraschungswerte eintragen."""
        prio = (np.abs(td_errors) + self.epsilon) ** self.alpha
        self.max_priority = max(self.max_priority, float(prio.max()))
        self.tree.set(idx, prio)


# =========================================================================== #
def make_buffer(cfg, state_size: int, rng: np.random.Generator | None = None):
    """Baut den in der Config gewaehlten Puffer-Typ."""
    if getattr(cfg, "prioritized", False):
        return PrioritizedReplayBuffer(
            cfg.buffer_size, state_size,
            alpha=cfg.per_alpha, beta_start=cfg.per_beta_start,
            beta_steps=cfg.per_beta_steps, rng=rng,
        )
    return ReplayBuffer(cfg.buffer_size, state_size, rng=rng)
