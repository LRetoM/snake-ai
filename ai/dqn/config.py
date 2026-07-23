"""Alle Stellschrauben des DQN-Trainings an EINEM Ort.

Warum eine eigene Config-Datei? Damit du beim Experimentieren nur hier drehst und
Agent/Trainer/Dashboard unberuehrt bleiben. Genau das war dein Wunsch: viele
Faktoren, die man verbessern (oder verschlechtern) kann, ohne dass wir gleich an
eine Grenze stossen.

Kurz-Glossar (fuer den KI-Einstieg):
- "Q-Wert": die geschaetzte Gesamt-Belohnung, die eine Aktion in einer Situation
  noch bringt. Das Netz gibt 3 davon aus (geradeaus/links/rechts).
- "gamma" (Diskont): wie stark zukuenftige Belohnung im Vergleich zu sofortiger
  zaehlt. 0.95 heisst: ein Punkt in 20 Zuegen ist ~0.36 heutige Punkte wert.
- "epsilon": Neugier. Wahrscheinlichkeit, einen ZUFALLS-Zug zu machen statt dem
  besten bekannten. Startet hoch (viel ausprobieren), sinkt langsam (mehr vertrauen).
- "Replay-Puffer": ein Tagebuch alter Erfahrungen, aus dem wiederholt gelernt wird.
- "Target-Netz": eine eingefrorene Kopie des Netzes als stabiles Lernziel.
- "n-Schritt": es wird nicht aus einzelnen Zuegen gelernt, sondern aus kurzen
  Ketten -- Wissen wandert dadurch schneller rueckwaerts.

Die Standardwerte hier sind NICHT geraten, sondern auf diesem Projekt gemessen
(siehe die Vergleichslaeufe im Chat / logs/). Wer experimentiert: immer nur EINEN
Wert auf einmal aendern, sonst weiss man hinterher nicht, was gewirkt hat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DQNConfig:
    # ------------------------------------------------------------------ #
    # Umgebung (identisch zum Menschen-Standard, damit der Vergleich fair ist)
    # ------------------------------------------------------------------ #
    grid_cols: int = 20
    grid_rows: int = 20
    # 3 statt 1 Frucht: mehr gleichzeitige Lernanlaesse pro Partie, war Teil
    # der besten bisher GEMESSENEN Kombination (Pruefung 67, Champion 75.6 --
    # siehe Kommentare bei hidden/gamma/n_step/eps_decay_steps unten).
    fruit_count: int = 3
    wrap_walls: bool = False

    # ------------------------------------------------------------------ #
    # Wahrnehmung -- was die KI ueberhaupt "sieht"
    # ------------------------------------------------------------------ #
    # "simple" = die 11 Zahlen, die auch die Neuroevolution benutzt.
    # "rich"   = 39 Zahlen: Abstand zu Wand/Koerper/Frucht in 8 Richtungen,
    #            Frucht-Versatz, Schwanzrichtung, Laenge, Hunger.
    # Der grosse Unterschied: mit "simple" sehen tausende verschiedene
    # Situationen fuer die KI voellig gleich aus -- sie kann ab einem gewissen
    # Punkt gar nicht mehr besser werden, egal wie lange sie trainiert.
    # Details und die Leitplanken dazu: ai/perception.py.
    perception: str = "rich"

    # ------------------------------------------------------------------ #
    # Das Gehirn (Netzgroesse)
    # ------------------------------------------------------------------ #
    # Eingaenge -> versteckte Schichten -> 3 Ausgaenge (Q-Werte).
    # Groesser = kann Kompliziertes lernen, kostet aber Rechenzeit pro Zug.
    # (256, 128) statt (128, 128): Teil der bisher besten GEMESSENEN Kombi
    # (Pruefung 67, Champion 75.6 -- deutlich vor den reinen Code-Defaults,
    # die nur 58 erreichten). Mehr Kapazitaet fuer die reichere Wahrnehmung.
    hidden: tuple[int, ...] = (256, 128)

    # ------------------------------------------------------------------ #
    # Die parallel laufenden Spiele
    # ------------------------------------------------------------------ #
    # So viele Schlangen spielen GLEICHZEITIG und fuellen EIN gemeinsames
    # Tagebuch, gesteuert von EINEM gemeinsamen Gehirn. Ein Fehler von Schlange 3
    # lehrt alle. Mehr Spiele = mehr frische, unterschiedliche Erfahrung pro
    # Lernschritt (und kaum Mehrkosten, weil alle in EINEM Netz-Durchlauf
    # entscheiden). 16 statt 8: Teil derselben gemessenen Bestkombi.
    num_games: int = 16

    # ------------------------------------------------------------------ #
    # Lernen (Gradientenabstieg mit PyTorch)
    # ------------------------------------------------------------------ #
    learning_rate: float = 1e-3     # wie grosse Lernschritte gemacht werden
    # 0.97 statt 0.95: weitsichtiger, Teil derselben gemessenen Bestkombi --
    # bei 3 Fruechten und laengeren Partien zahlt sich mehr Weitsicht aus.
    gamma: float = 0.97
    # Laenge der Erfahrungs-Ketten. 1 = klassisch (jeder Zug lernt von seinem
    # direkten Nachfolger). In einem ISOLIERTEN A/B-Test (alte Defaults sonst
    # unveraendert) brachte n_step=3 nichts (27.4 statt 26.3, Rauschen) --
    # ABER in Kombination mit den anderen Werten hier (mehr Spiele, groesseres
    # Netz, mehr Fruechte, mehr Weitsicht) war genau diese Gesamt-Kombination
    # die bisher beste gemessene (Pruefung 67, Champion 75.6). Deshalb hier
    # uebernommen, obwohl der isolierte Effekt von n_step allein unklar bleibt.
    n_step: int = 3
    batch_size: int = 256           # so viele Tagebuch-Eintraege pro Lernschritt
    buffer_size: int = 100_000      # Groesse des Tagebuchs (aelteste fallen raus)
    min_buffer: int = 2_000         # erst lernen, wenn so viel Erfahrung da ist
    train_every: int = 1            # nur bei jedem N-ten Tick lernen (2 = doppelter
                                    # Spiel-Durchsatz, halb so viele Lernschritte)
    train_iters_per_step: int = 1   # Lernschritte pro Lern-Tick
    target_update: int = 1_000      # alle N Lernschritte Target-Netz aktualisieren
    grad_clip: float = 10.0         # Gradienten kappen -> stabileres Lernen
    # Double DQN: das lernende Netz WAEHLT die beste Folge-Aktion, das eingefrorene
    # Target-Netz BEWERTET sie. Zwei getrennte Meinungen bremsen die bekannte
    # Neigung von DQN, sich selbst zu ueberschaetzen. Kostet nichts, hilft meist.
    double_dqn: bool = True

    # ------------------------------------------------------------------ #
    # Priorisiertes Tagebuch (Prioritized Experience Replay)
    # ------------------------------------------------------------------ #
    # An: Erinnerungen, bei denen sich das Netz stark verschaetzt hat, werden
    # oefter wiederholt -- wie Karteikarten, die man falsch hatte.
    # STANDARDMAESSIG AUS, und das ist ein Messergebnis, keine Meinung: im
    # Vergleichslauf bei gleichem Zeitbudget war die Version MIT Priorisierung
    # schlechter (Pruefung 30.5 statt 42.8) UND ein Drittel langsamer. Grund:
    # unsere Belohnung ist bereits sehr dicht (jeder Zug gibt Rueckmeldung) --
    # PER glaenzt vor allem dort, wo Belohnung extrem selten ist. Der Code ist
    # da und getestet; bei spaeteren Aenderungen (z.B. Belohnung nur noch fuer
    # Fruechte) lohnt sich ein neuer Vergleich. Details: ai/dqn/memory.py.
    prioritized: bool = False
    per_alpha: float = 0.6          # wie stark priorisiert wird (0 = gar nicht)
    per_beta_start: float = 0.4     # Staerke der Gegenkorrektur am Anfang
    per_beta_steps: int = 200_000   # ... waechst ueber so viele Lernschritte auf 1.0

    # ------------------------------------------------------------------ #
    # Neugier (Epsilon-greedy)
    # ------------------------------------------------------------------ #
    eps_start: float = 1.0          # Anfang: 100% Zufall (reines Ausprobieren)
    eps_end: float = 0.02           # Ende: fast immer der beste bekannte Zug
    # 80k statt 40k: Teil derselben gemessenen Bestkombi -- laenger ausprobieren,
    # bevor die Neugier abgeschaltet wird, zahlt sich mit mehr Spielen/groesserem
    # Netz aus.
    eps_decay_steps: int = 80_000   # linear von start->end ueber so viele Ticks

    # ------------------------------------------------------------------ #
    # Belohnung = das Lernsignal (LEBT HIER, NICHT IM SPIEL!)
    # ------------------------------------------------------------------ #
    # Leitplanke: das ist Feedback ("das war gut/schlecht"), keine Strategie.
    # Die KI erfaehrt nur, DASS Frucht gut ist -- nicht WIE sie hinkommt.
    reward_fruit: float = 10.0      # Frucht gefressen -> starke Belohnung
    reward_death: float = -10.0     # gestorben (Wand/Selbst/Verhungern) -> Strafe
    reward_step: float = -0.01      # winzige Zeitstrafe je Zug (gegen Troedeln)
    reward_closer: float = 0.1      # naeher an die naechste Frucht gekommen
    reward_farther: float = -0.12   # weiter weg (leicht haerter -> gegen Kreisrennen)

    # ------------------------------------------------------------------ #
    # Verhungern (Trainings-Timeout, KEINE Spielregel)
    # ------------------------------------------------------------------ #
    # Laeuft die Schlange zu lange ohne Frucht, brechen wir die Partie ab (mit
    # Todesstrafe) -- sonst koennte sie ewig sichere Kreise drehen. Das Limit
    # WAECHST mit der Laenge: eine lange Schlange darf laenger suchen.
    starve_base: int = 100
    starve_growth: int = 20         # Limit = starve_base + starve_growth * Laenge

    # ------------------------------------------------------------------ #
    # Pruefung ("wie gut ist die KI WIRKLICH?")
    # ------------------------------------------------------------------ #
    # Der Ø-Score im Training ist geschoent bzw. verzerrt: dort wuerfelt die KI
    # noch bei jedem epsilon-ten Zug. Bei langen Partien (300+ Zuege) sind das
    # mehrere Zufallszuege pro Partie -- und ein einziger davon kann eine lange
    # Schlange toeten. Deshalb spielt sie regelmaessig ein paar Partien OHNE
    # Zufall (epsilon = 0). Dieser Pruefungs-Score ist der ehrliche Wert -- und
    # genau der, den du auch beim Zuschauen (watch_ai.py) siehst.
    eval_every_episodes: int = 200  # alle N Episoden eine Pruefung
    eval_episodes: int = 10         # so viele Partien pro Pruefung
    eval_max_steps: int = 4_000     # Notbremse, falls eine Pruefpartie ewig laeuft

    # ------------------------------------------------------------------ #
    # Technik / Geschwindigkeit
    # ------------------------------------------------------------------ #
    # PyTorch-CPU-Threads. 0 = beim Start EINMAL ausmessen, welche Anzahl auf
    # DIESEM Rechner am schnellsten ist (dauert unter einer Sekunde). Das ist
    # der groesste Geschwindigkeits-Hebel ueberhaupt und faellt je nach Rechner
    # voellig unterschiedlich aus -- deshalb messen statt raten. Eine feste Zahl
    # (1, 2, 4 ...) erzwingt stattdessen genau diesen Wert.
    torch_threads: int = 0
    seed: int | None = None         # None = jeder Lauf anders

    def starve_limit(self, length: int) -> int:
        """Erlaubte Schritte ohne Frucht, bevor die Partie als 'verhungert' endet."""
        return self.starve_base + self.starve_growth * length
