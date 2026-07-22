"""Startbefehl fuer das Deep-Q-Learning-Training (die ZWEITE, eigenstaendige KI).

Start:
    source venv/bin/activate      (Mac)   /   venv\\Scripts\\activate   (Windows)
    python train_dqn.py

Wichtig: Das hier MUSS unter dem normalen CPython-venv laufen, nicht unter PyPy --
DQN rechnet mit PyTorch, und fuer PyPy gibt es kein torch. (Der PyPy-Umweg gilt
nur fuer die Neuroevolution, die kommt mit reinem NumPy aus.)

Es oeffnet sich ein Fenster mit:
  - den parallel spielenden Schlangen (live, waehrend sie lernen)
  - Kennzahlen: Ø-Score, Bestwert, Neugier, Lernfehler, Tagebuch-Fuellstand ...
  - den Todesursachen und der Lernkurve

Steuerung im Fenster:
    Leertaste = Pause    ←/→ oder +/- = Geschwindigkeit
    T = Turbo (max. Lerntempo, Felder werden nicht gezeichnet)
    Esc = zurueck ins Menue

Alle Stellschrauben (Belohnungen, Puffergroesse, Batch, Verhungern-Limit ...)
stehen in ai/dqn/config.py. Die wichtigsten davon lassen sich auch direkt im
Startmenue des Fensters umstellen.

Der jeweils beste Bot wird automatisch nach models/dqn_champion.pt gespeichert.
"""

from __future__ import annotations

from ai.dqn.config import DQNConfig
from dashboard.dqn_view import main as run_dashboard


def main() -> None:
    # Hier koennte man Vorgaben setzen, die das Menue dann uebernimmt --
    # z.B. DQNConfig(num_games=5, batch_size=512). Ohne Argumente gelten die
    # Standardwerte aus ai/dqn/config.py.
    run_dashboard(DQNConfig())


if __name__ == "__main__":
    main()
