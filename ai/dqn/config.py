"""Alle Stellschrauben des DQN-Trainings an EINEM Ort.

Warum eine eigene Config-Datei? Damit du beim Experimentieren nur hier drehst und
Agent/Trainer/Dashboard unberuehrt bleiben. Genau das war dein Wunsch: viele
Faktoren, die man verbessern (oder verschlechtern) kann, ohne dass wir gleich an
eine Grenze stossen.

Kurz-Glossar (fuer den KI-Einstieg):
- "Q-Wert": die geschaetzte Gesamt-Belohnung, die eine Aktion in einer Situation
  noch bringt. Das Netz gibt 3 davon aus (geradeaus/links/rechts).
- "gamma" (Diskont): wie stark zukuenftige Belohnung im Vergleich zu sofortiger
  zaehlt. 0.9 heisst: ein Punkt in 10 Zuegen ist ~0.35 heutige Punkte wert.
- "epsilon": Neugier. Wahrscheinlichkeit, einen ZUFALLS-Zug zu machen statt dem
  besten bekannten. Startet hoch (viel ausprobieren), sinkt langsam (mehr vertrauen).
- "Replay-Puffer": ein Tagebuch alter Erfahrungen, aus dem wiederholt gelernt wird.
- "Target-Netz": eine eingefrorene Kopie des Netzes als stabiles Lernziel.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    # --- Umgebung (identisch zum Menschen-Standard, damit der Vergleich fair ist) ---
    grid_cols: int = 20
    grid_rows: int = 20
    fruit_count: int = 1
    wrap_walls: bool = False

    # --- Das Gehirn (Netzgroesse) ---
    # 11 Eingaenge -> zwei versteckte Schichten -> 3 Ausgaenge (Q-Werte).
    # Groesser = kann Kompliziertes lernen, aber langsamer pro Zug. (128,128) ist
    # ein guter Kompromiss fuer die 11 Wahrnehmungszahlen.
    hidden: tuple[int, ...] = (128, 128)

    # --- Die 5 verbundenen Spiele ---
    # So viele Schlangen spielen GLEICHZEITIG und fuellen EIN gemeinsames Tagebuch,
    # gesteuert von EINEM gemeinsamen Gehirn. Ein Fehler von Schlange 3 lehrt alle 5.
    num_games: int = 5

    # --- Lernen (Gradientenabstieg mit PyTorch) ---
    learning_rate: float = 1e-3     # wie grosse Lernschritte gemacht werden
    gamma: float = 0.9              # Diskont fuer zukuenftige Belohnung
    batch_size: int = 256           # so viele Tagebuch-Eintraege pro Lernschritt
    buffer_size: int = 100_000      # Groesse des Tagebuchs (aelteste fallen raus)
    min_buffer: int = 2_000         # erst lernen, wenn so viel Erfahrung da ist
    train_iters_per_step: int = 1   # Lernschritte pro Umwelt-Tick (aller Spiele)
    target_update: int = 1_000      # alle N Lernschritte Target-Netz aktualisieren
    grad_clip: float = 10.0         # Gradienten kappen -> stabileres Lernen
    # Double DQN: das lernende Netz WAEHLT die beste Folge-Aktion, das eingefrorene
    # Target-Netz BEWERTET sie. Zwei getrennte Meinungen bremsen die bekannte
    # Neigung von DQN, sich selbst zu ueberschaetzen. Kostet nichts, hilft meist.
    double_dqn: bool = True

    # --- Neugier (Epsilon-greedy) ---
    eps_start: float = 1.0          # Anfang: 100% Zufall (reines Ausprobieren)
    eps_end: float = 0.02           # Ende: fast immer der beste bekannte Zug
    eps_decay_steps: int = 40_000   # linear von start->end ueber so viele Umwelt-Schritte

    # --- Belohnung = das Lernsignal (LEBT HIER, NICHT IM SPIEL!) ---
    # Leitplanke: das ist Feedback ("das war gut/schlecht"), keine Strategie.
    # Die KI erfaehrt nur, DASS Frucht gut ist -- nicht WIE sie hinkommt.
    reward_fruit: float = 10.0      # Frucht gefressen -> starke Belohnung
    reward_death: float = -10.0     # gestorben (Wand/Selbst/Verhungern) -> Strafe
    reward_step: float = -0.01      # winzige Zeitstrafe je Zug (gegen Troedeln)
    reward_closer: float = 0.1      # naeher an die naechste Frucht gekommen
    reward_farther: float = -0.12   # weiter weg (leicht haerter -> gegen Kreisrennen)

    # --- Verhungern (Trainings-Timeout, KEINE Spielregel) ---
    # Laeuft die Schlange zu lange ohne Frucht, brechen wir die Partie ab (mit
    # Todesstrafe) -- sonst koennte sie ewig sichere Kreise drehen. Das Limit
    # WAECHST mit der Laenge: eine lange Schlange darf laenger suchen.
    starve_base: int = 100
    starve_growth: int = 20         # Limit = starve_base + starve_growth * Laenge

    def starve_limit(self, length: int) -> int:
        """Erlaubte Schritte ohne Frucht, bevor die Partie als 'verhungert' endet."""
        return self.starve_base + self.starve_growth * length
