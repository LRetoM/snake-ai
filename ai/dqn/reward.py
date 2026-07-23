"""Die Belohnung -- das Lernsignal der DQN-KI. Reines FEEDBACK, keine Strategie.

Leitplanke (CLAUDE.md), bitte im Kopf behalten:
Diese Datei lebt bewusst im KI-Ordner und NICHT im Spiel. Das Spiel
(game/snake_game.py) kennt keine Belohnungen -- es kennt nur Punkte, Tod und
Sieg. Belohnung ist eine reine Trainings-Erfindung: unsere Art, der KI "gut
gemacht" oder "schlecht" zu sagen. Wuerde man sie ins Spiel schreiben, waere die
Trennung Umgebung <-> Agent kaputt.

Genauso wichtig: Belohnung sagt der KI nur, WAS erstrebenswert ist -- nie WIE
sie es erreicht. Wir verraten ihr nicht "geh nach rechts" oder "meide Sackgassen".
Wir sagen nur: Frucht = gut, Tod = schlecht, Troedeln = leicht schlecht, sich der
Frucht naehern = ein bisschen gut. Wie man das anstellt (Koerper umfahren,
Sackgassen vermeiden, sich Platz lassen), muss die KI vollstaendig selbst
herausfinden. Kein Pathfinding, keine Heuristik.

Die Werte (alle einstellbar in ai/dqn/config.py)
-----------------------------------------------
    Frucht gefressen                     +10
    gestorben (Wand/Selbst/Verhungern)   -10
    pro Zug                              -0.01
    naeher an die naechste Frucht        +0.10
    weiter weg von der Frucht            -0.12

Warum ueberhaupt "naeher/weiter"? (Potenzialbasierte Formung)
-------------------------------------------------------------
Ohne diesen Hinweis bekaeme die KI am Anfang fast nur Nullen: sie stolpert
zufaellig herum, trifft praktisch nie eine Frucht und hat damit kaum etwas zu
lernen -- wie jemand, dem man beim Vokabeltraining erst nach 500 Karten sagt, ob
er richtig lag. Der Naeher/Weiter-Hinweis gibt in JEDEM Zug eine kleine
Rueckmeldung und beschleunigt den Start enorm.

Der schoene Teil: dieses Muster (Belohnung anhand der VERAENDERUNG einer
Abstandsgroesse) heisst *potential-based reward shaping*. Dafuer gibt es einen
mathematischen Beweis (Ng, Harada & Russell, 1999): es aendert die OPTIMALE
Strategie nicht -- es macht nur den Weg dorthin schneller. Es ist also ein
Wegweiser, kein Betrug.

"weiter weg" (-0.12) ist absichtlich etwas haerter als "naeher" (+0.10).
Grund: waeren beide gleich gross, koennte die Schlange endlos hin- und herpendeln
(einmal hin +0.10, einmal zurueck -0.10, macht netto 0) und trotzdem "okay"
dastehen. Mit der Asymmetrie kostet jedes Pendeln unterm Strich etwas -- Kreise
drehen lohnt sich nicht mehr.

Distanz = Manhattan-Distanz |dx| + |dy|: die Anzahl Schritte, die man auf einem
Gitter mindestens braucht (Diagonalen gibt es ja nicht). Sie ist bewusst
"dumm" -- sie geht durch Waende und durch den eigenen Koerper hindurch. Sie ist
damit ein Richtungs-GEFUEHL ("da drueben liegt was"), keine Wegplanung.
"""

from __future__ import annotations

from game.snake_game import SnakeGame


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Gitter-Abstand zwischen zwei Zellen: |dx| + |dy|."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def fruit_distance(game: SnakeGame) -> int:
    """Manhattan-Distanz vom Kopf zur NAECHSTEN Frucht.

    Liegt (theoretisch) gar keine Frucht mehr auf dem Feld -- das passiert nur,
    wenn das Feld praktisch voll ist, also kurz vor dem Sieg -- geben wir 0
    zurueck, damit die Rechnung nicht abstuerzt.
    """
    if not game.fruits:
        return 0
    head = game.head
    return min(manhattan(head, fruit) for fruit in game.fruits)


def compute_reward(cfg, ate_fruit: bool, died: bool,
                   dist_before: int, dist_after: int,
                   won: bool = False, formung_faktor: float = 1.0) -> float:
    """Belohnung fuer EINEN Zug.

    Argumente:
      cfg          -- DQNConfig (enthaelt alle Belohnungswerte)
      ate_fruit    -- wurde in diesem Zug eine Frucht gefressen?
      died         -- ist die Partie durch diesen Zug zu Ende (Wand, Selbst,
                      Verhungern)? Verhungern entscheidet der Trainer, nicht
                      das Spiel -- siehe trainer.py.
      dist_before  -- Fruchtabstand VOR dem Zug
      dist_after   -- Fruchtabstand NACH dem Zug
      won          -- wurde das FELD KOMPLETT gefuellt (siehe TRAININGSPLAN.md
                      2.4)? Gibt zusaetzlich zur Frucht-Belohnung einen
                      grossen Sieg-Bonus -- ohne den gibt das Vollmachen des
                      Feldes nur so viel wie jede andere Frucht, obwohl es
                      das eigentliche Ziel ist.
      formung_faktor -- 1.0 = volle Naeher/Weiter-Formung (hilft dem
                      Fruehspiel-Start), 0.0 = komplett ausgeblendet. Faellt
                      mit steigendem Pruefungs-Niveau linear auf 0 (siehe
                      MultiGameTrainer.formung_faktor): die Formung belohnt
                      den KUERZESTEN Weg zur Frucht -- im Fruehspiel richtig,
                      im Endspiel oft genau der Weg in die Selbst-Falle.

    Reihenfolge der Faelle ist Absicht:

    1) Tod schlaegt alles. Die Schlange ist nicht mehr auf dem Feld, ein
       "Abstand zur Frucht" waere sinnlos. Es gibt genau die Todesstrafe.
    2) Frucht gefressen: volle +10 (plus die winzige Zeitstrafe, plus
       Sieg-Bonus falls das Feld dabei komplett voll wurde). Hier gibt es
       BEWUSST keinen Naeher/Weiter-Anteil -- die Frucht ist ja weg und die
       naechste irgendwo neu gespawnt, der Abstand wuerde also zufaellig
       springen. Dieser Zufall waere reines Rauschen im Lernsignal.
    3) Normaler Zug: Zeitstrafe + (abklingender) Naeher/Weiter-Hinweis.
    """
    if died:
        return cfg.reward_death

    reward = cfg.reward_step  # winzige Zeitstrafe, gilt fuer jeden Zug

    if ate_fruit:
        reward += cfg.reward_fruit
        if won:
            reward += cfg.reward_win
        return reward

    if dist_after < dist_before:
        reward += formung_faktor * cfg.reward_closer
    elif dist_after > dist_before:
        reward += formung_faktor * cfg.reward_farther
    # gleich geblieben (kann bei gleichem Abstand um die Ecke passieren):
    # nur die Zeitstrafe, kein Bonus, keine Extra-Strafe.

    return reward
