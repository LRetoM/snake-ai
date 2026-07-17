"""Reine Snake-Spiellogik -- OHNE pygame, ohne Grafik, ohne Tastatur.

Das ist die "Umgebung" (Environment). Sie legt ALLE Regeln fest:
Bewegung, Kollision, wo Fruechte spawnen, wann man stirbt. Grafik und Eingabe
passieren woanders (renderer.py / play_human.py).

Warum diese strikte Trennung? Spaeter soll die KI dieselbe Logik ohne Fenster
tausende Male pro Sekunde durchrechnen. Deshalb darf hier nichts sein, das ein
sichtbares Fenster braucht. Die KI bekommt spaeter nur:
  - eine Wahrnehmung des Zustands (perception.py, kommt in Schritt 2)
  - eine Aktion (geradeaus / links / rechts)
  - Feedback (Punkte / Tod)
... aber niemals Zugriff auf die Regeln selbst.

Analogie: Das Spiel ist das "Brett mit Regeln". Ein Mensch (oder die KI) sieht
nur das Brett und darf Zuege machen -- die Regeln aendern kann er nie.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """Die vier Bewegungsrichtungen als (dx, dy)-Vektor.

    dx = Aenderung der Spalte, dy = Aenderung der Zeile.
    Achtung: y zeigt nach UNTEN (wie bei Bildschirmkoordinaten ueblich),
    deshalb ist UP = (0, -1).
    """
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def opposite(self) -> "Direction":
        """Gegenrichtung -- gebraucht, um 180-Grad-Wenden zu verbieten."""
        dx, dy = self.value
        return Direction((-dx, -dy))


class Action(Enum):
    """Die drei Aktionen, die die KI waehlen kann -- RELATIV zur Blickrichtung.

    Das ist die einzige Sprache, in der die KI mit dem Spiel spricht. Sie denkt
    nicht in Himmelsrichtungen ("nach oben"), sondern wie ein Autofahrer:
    geradeaus, links abbiegen oder rechts abbiegen. Dadurch gibt es hier auch
    gar keine 180-Grad-Wende -- ein Selbstmord-Zug ist strukturell unmoeglich.
    """
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


# Die vier Richtungen im Uhrzeigersinn. Damit lassen sich relative Drehungen
# ("rechts" = ein Schritt im Uhrzeigersinn) einfach ausrechnen.
_CLOCKWISE = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]


def relative_turn(direction: "Direction", action: "Action") -> "Direction":
    """Rechnet eine relative Aktion in eine absolute Richtung um.

    Beispiel: schaut die Schlange nach oben (UP) und die Aktion ist RIGHT,
    dann ist die neue Richtung RIGHT (rechts). Schaut sie nach RIGHT und die
    Aktion ist RIGHT, wird daraus DOWN -- usw. im Uhrzeigersinn.
    """
    i = _CLOCKWISE.index(direction)
    if action == Action.RIGHT:
        i = (i + 1) % 4        # ein Schritt im Uhrzeigersinn
    elif action == Action.LEFT:
        i = (i - 1) % 4        # ein Schritt gegen den Uhrzeigersinn
    return _CLOCKWISE[i]       # STRAIGHT -> unveraendert


# Eine Position auf dem Gitter: (spalte, zeile).
Cell = tuple[int, int]


@dataclass
class StepResult:
    """Was bei einem Spielschritt passiert ist -- spaeter das Lernsignal der KI.

    - alive:      lebt die Schlange nach diesem Schritt noch?
    - ate_fruit:  wurde in diesem Schritt eine Frucht gefressen?
    - won:        ist das Spielfeld komplett gefuellt (Sieg)?
    - score:      aktueller Punktestand
    """
    alive: bool
    ate_fruit: bool
    won: bool
    score: int


class SnakeGame:
    """Der Spielzustand + die Regeln. Kein Rendering, keine Eingabe.

    Typischer Ablauf:
        game = SnakeGame(config)
        game.reset()
        game.change_direction(Direction.UP)   # Wunschrichtung setzen
        result = game.step()                   # einen Schritt weiterrechnen
    """

    def __init__(self, config, rng: random.Random | None = None) -> None:
        # config hat u.a. grid_cols, grid_rows, fruit_count, wrap_walls.
        self.cols = config.grid_cols
        self.rows = config.grid_rows
        self.fruit_count = max(1, config.fruit_count)
        self.wrap_walls = config.wrap_walls

        # Eigener Zufallsgenerator -> spaeter reproduzierbar (Seed) fuers Training.
        self.rng = rng or random.Random()

        # Felder werden in reset() gefuellt; hier nur angekuendigt.
        self.snake: list[Cell] = []
        self.direction: Direction = Direction.RIGHT
        self.fruits: set[Cell] = set()
        self.score: int = 0
        self.steps: int = 0             # Anzahl gemachter Schritte (Ueberlebenszeit)
        self.steps_since_fruit: int = 0  # gegen "im Kreis laufen" (spaeter fuer KI)
        self.alive: bool = True
        self.won: bool = False

        # Gepufferter naechster Zug (hoechstens EINER). Ein neuer Tastendruck
        # ueberschreibt einen bereits gepufferten Zug komplett -- so wirkt die
        # Steuerung so reaktionsschnell wie moeglich: der zuletzt gedrueckte
        # gueltige Zug gewinnt immer, es gibt keine Warteschlange aus mehreren
        # Zuegen, die erst nacheinander abgearbeitet werden muesste.
        self._pending_turn: Direction | None = None

        self.reset()

    # ------------------------------------------------------------------ #
    # Auf- und Zuruecksetzen
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Setzt eine frische Runde auf: Schlange in der Mitte, Fruechte gespawnt."""
        mid_x = self.cols // 2
        mid_y = self.rows // 2

        # Schlange der Laenge 3, waagerecht, Kopf rechts. Reihenfolge:
        # snake[0] = Kopf, snake[-1] = Schwanzende.
        self.snake = [
            (mid_x, mid_y),
            (mid_x - 1, mid_y),
            (mid_x - 2, mid_y),
        ]
        self.direction = Direction.RIGHT
        self._pending_turn = None

        self.score = 0
        self.steps = 0
        self.steps_since_fruit = 0
        self.alive = True
        self.won = False

        self.fruits = set()
        self._refill_fruits()

    # ------------------------------------------------------------------ #
    # Eingabe: Wunschrichtung setzen
    # ------------------------------------------------------------------ #
    def change_direction(self, new_direction: Direction) -> None:
        """Legt die naechste Richtung fest (fuer menschliche Steuerung).

        Die Richtung wird gepuffert und erst beim naechsten step() angewandt
        (verhindert 180-Grad-Selbstmord: die Pruefung erfolgt gegen die aktuell
        wirksame Richtung, nicht gegen einen erst geplanten Zug).

        Ein neuer Tastendruck ERSETZT einen bereits gepufferten Zug vollstaendig.
        Das ist bewusst so (nicht als Warteschlange von mehreren Zuegen), damit
        die Steuerung maximal reaktionsschnell bleibt: Aendert der Spieler
        innerhalb eines Schritts seine Meinung, zaehlt immer der zuletzt
        gedrueckte gueltige Zug -- er muss nicht erst einen aelteren, ueberholten
        Zug abwarten.
        """
        # Immer gegen die AKTUELLE (nicht eine gepufferte) Richtung pruefen.
        if new_direction == self.direction or new_direction == self.direction.opposite:
            return
        self._pending_turn = new_direction

    # ------------------------------------------------------------------ #
    # Ein Spielschritt -- KI-Variante (relative Aktion)
    # ------------------------------------------------------------------ #
    def step_action(self, action: Action) -> StepResult:
        """Fuehrt EINEN Schritt mit einer relativen Aktion aus (fuer die KI).

        Waehrend `change_direction()` + `step()` fuer die Menschsteuerung gedacht
        sind (absolute Richtung, gepuffert), setzt die KI ihre Richtung direkt
        ueber geradeaus/links/rechts. Ein 180-Grad-Selbstmord ist damit unmoeglich.
        """
        self.direction = relative_turn(self.direction, action)
        self._pending_turn = None  # evtl. gepufferte Menscheneingabe verwerfen
        return self.step()

    # ------------------------------------------------------------------ #
    # Ein Spielschritt
    # ------------------------------------------------------------------ #
    def step(self) -> StepResult:
        """Rechnet die Schlange genau einen Schritt weiter und wendet Regeln an."""
        if not self.alive or self.won:
            return StepResult(self.alive, False, self.won, self.score)

        # 1) Gepufferten Zug uebernehmen (falls vorhanden).
        if self._pending_turn is not None:
            self.direction = self._pending_turn
            self._pending_turn = None

        # 2) Neue Kopfposition berechnen.
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_x, new_y = head_x + dx, head_y + dy

        # 3) Waende behandeln.
        if self.wrap_walls:
            # Durchgang: modulo laesst die Schlange auf der anderen Seite rauskommen.
            new_x %= self.cols
            new_y %= self.rows
        else:
            # Klassisch: ausserhalb des Feldes = Tod.
            if not (0 <= new_x < self.cols and 0 <= new_y < self.rows):
                self.alive = False
                return StepResult(False, False, False, self.score)

        new_head: Cell = (new_x, new_y)

        # 4) Wird gefressen? Dann waechst die Schlange (Schwanz bleibt).
        will_eat = new_head in self.fruits

        # 5) Selbstkollision pruefen.
        #    Wenn NICHT gefressen wird, rueckt der Schwanz eine Zelle weiter --
        #    die aktuelle Schwanzzelle wird also frei und darf betreten werden.
        occupied = set(self.snake)
        if not will_eat:
            occupied.discard(self.snake[-1])
        if new_head in occupied:
            self.alive = False
            return StepResult(False, False, False, self.score)

        # 6) Bewegen: neuen Kopf vorne einfuegen.
        self.snake.insert(0, new_head)

        self.steps += 1
        ate_fruit = False

        if will_eat:
            # Frucht gefressen: Punkt gutschreiben, Frucht entfernen, neue nachlegen.
            self.fruits.discard(new_head)
            self.score += 1
            self.steps_since_fruit = 0
            ate_fruit = True
            self._refill_fruits()
        else:
            # Nicht gefressen: Schwanzende entfernen -> Schlange bleibt gleich lang.
            self.snake.pop()
            self.steps_since_fruit += 1

        # 7) Sieg? Wenn kein freies Feld mehr existiert, ist alles voll -> gewonnen.
        if len(self.snake) >= self.cols * self.rows:
            self.won = True
            return StepResult(True, ate_fruit, True, self.score)

        return StepResult(True, ate_fruit, False, self.score)

    # ------------------------------------------------------------------ #
    # Fruechte
    # ------------------------------------------------------------------ #
    def _refill_fruits(self) -> None:
        """Legt so viele Fruechte nach, bis wieder fruit_count auf dem Feld sind.

        Fruechte spawnen NUR auf freien Zellen (nicht auf der Schlange und nicht
        auf einer bereits liegenden Frucht). Gibt es keine freie Zelle mehr, wird
        einfach keine gelegt (dann ist das Feld praktisch voll -> Sieg naht).
        """
        blocked = set(self.snake) | self.fruits
        free_cells = [
            (x, y)
            for x in range(self.cols)
            for y in range(self.rows)
            if (x, y) not in blocked
        ]

        while len(self.fruits) < self.fruit_count and free_cells:
            choice = self.rng.choice(free_cells)
            free_cells.remove(choice)
            self.fruits.add(choice)

    # ------------------------------------------------------------------ #
    # Praktische Helfer (fuer Renderer & spaeter fuer die Wahrnehmung der KI)
    # ------------------------------------------------------------------ #
    @property
    def head(self) -> Cell:
        """Aktuelle Kopfposition."""
        return self.snake[0]

    @property
    def length(self) -> int:
        """Aktuelle Laenge der Schlange (= Anzahl Segmente)."""
        return len(self.snake)

    def is_cell_free(self, cell: Cell) -> bool:
        """True, wenn die Zelle innerhalb des Feldes und nicht von der Schlange belegt ist."""
        x, y = cell
        if not (0 <= x < self.cols and 0 <= y < self.rows):
            return False
        return cell not in set(self.snake)

    def is_deadly(self, cell: Cell) -> bool:
        """True, wenn ein Zug auf diese Zelle die Schlange toeten wuerde.

        Das ist ein reiner Wahrnehmungs-Helfer fuer perception.py: Er spiegelt
        exakt die Todesregel aus step() wider (Wand ODER Koerper -- ausser dem
        Schwanzende, das im naechsten Schritt sowieso wegrueckt). Die KI SIEHT
        damit nur "da vorne ist Gefahr" -- so wie ein Mensch die Wand sieht.
        Sie erfaehrt NICHT, was sie tun soll; das muss sie selbst lernen.
        """
        x, y = cell
        if self.wrap_walls:
            # Durchgang: eine Wand kann man nicht "treffen", man kommt hindurch.
            cell = (x % self.cols, y % self.rows)
        else:
            if not (0 <= x < self.cols and 0 <= y < self.rows):
                return True  # ausserhalb des Feldes = toedlich

        # Koerper pruefen; das Schwanzende wird als frei behandelt (rueckt weg).
        body = set(self.snake)
        body.discard(self.snake[-1])
        return cell in body
