"""Startet das Neuroevolution-Training mit Live-Dashboard.

    source venv/bin/activate      (Mac)   /   venv\\Scripts\\activate   (Windows)
    python train_evolution.py

Ein eigenes Fenster oeffnet sich: erst ein kleines Einstellungsmenue
(Populationsgroesse, Mutation, Fruechte, Anzahl sichtbarer Schlangen), dann
laeuft das Training live. Steuerung im Training: Leertaste = Pause, T = Turbo,
Pfeil hoch/runter = Anzeige-Geschwindigkeit, Esc = zurueck ins Menue.

Reiner Headless-Lauf ohne Fenster (nur Konsole, maximale Geschwindigkeit):
    python -c "from ai.evolution.train_evolution import main; main(generations=100)"
"""

from dashboard.live_view import main

if __name__ == "__main__":
    main()
