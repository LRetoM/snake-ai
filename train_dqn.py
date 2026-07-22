"""Startbefehl fuer das Deep-Q-Learning-Training (die ZWEITE, eigenstaendige KI).

Start (mit Fenster, zum Zusehen):
    source venv/bin/activate      (Mac)   /   venv\\Scripts\\activate   (Windows)
    python train_dqn.py

Start OHNE Fenster (maximaler Lern-Durchsatz, z.B. ueber Nacht):
    python train_dqn.py --headless 30      (30 Minuten rechnen)
    python train_dqn.py --headless 30 --weiter   (auf dem Champion aufbauen)

Der Unterschied: ohne Fenster faellt das Zeichnen komplett weg -- je nach
Rechner sind das nochmal 10-30% mehr Zuege pro Sekunde als der Turbo-Modus im
Fenster. Der Fortschritt wird stattdessen alle 30 Sekunden in die Konsole
geschrieben, und der beste Bot landet wie immer in models/dqn_champion.pt.

Wichtig: Das MUSS unter dem normalen CPython-venv laufen, nicht unter PyPy --
DQN rechnet mit PyTorch, und fuer PyPy gibt es kein torch. (Der PyPy-Umweg gilt
nur fuer die Neuroevolution, die kommt mit reinem NumPy aus.)

Steuerung im Fenster:
    Leertaste = Pause    ←/→ oder +/- = Geschwindigkeit
    T = Turbo (max. Lerntempo, Felder werden nicht gezeichnet)
    P = sofort eine Pruefung laufen lassen
    Esc = zurueck ins Menue

Zuschauen, wie gut der fertige Bot wirklich spielt:
    python watch_ai.py dqn

Alle Stellschrauben stehen in ai/dqn/config.py; die wichtigsten lassen sich
auch direkt im Startmenue des Fensters umstellen.
"""

from __future__ import annotations

import argparse
import time

from ai.dqn.config import DQNConfig


def run_headless(minutes: float, resume: bool) -> None:
    """Trainiert ohne jede Grafik und meldet den Fortschritt in der Konsole."""
    from ai.dqn.trainer import CHAMPION_PATH, MultiGameTrainer
    import os

    cfg = DQNConfig()
    resume_path = CHAMPION_PATH if (resume and os.path.exists(CHAMPION_PATH)) else None
    trainer = MultiGameTrainer(cfg, log_to_csv=True, resume_from=resume_path)

    print(f"Wahrnehmung: {cfg.perception} ({trainer.input_size} Werte)   "
          f"Netz: {cfg.hidden}   Spiele: {cfg.num_games}   "
          f"Threads: {trainer.agent.threads}")
    if resume_path:
        print(f"Weitertrainiert auf: {resume_path}")
    print(f"Laeuft {minutes:g} Minuten. Abbrechen mit Strg+C.\n")

    t_end = time.perf_counter() + minutes * 60
    next_report = time.perf_counter() + 30
    last_moves, last_t = 0, time.perf_counter()
    try:
        while time.perf_counter() < t_end:
            for _ in range(500):
                trainer.step()
            now = time.perf_counter()
            if now >= next_report:
                next_report = now + 30
                s = trainer.stats()
                rate = (s.total_moves - last_moves) / (now - last_t)
                last_moves, last_t = s.total_moves, now
                pruefung = "—" if s.eval_score is None else f"{s.eval_score:5.1f}"
                print(f"{int(s.elapsed) // 60:3d}:{int(s.elapsed) % 60:02d}  "
                      f"Ep {s.total_episodes:6d}  "
                      f"Pruefung {pruefung}  (best {s.eval_best:5.1f})  "
                      f"Training Ø {s.mean_score:5.1f}  "
                      f"eps {s.epsilon:.3f}  "
                      f"{rate:6.0f} Zuege/s", flush=True)
    except KeyboardInterrupt:
        print("\nAbgebrochen.")

    print("\nAbschluss-Pruefung laeuft ...")
    trainer.cfg.eval_episodes = 30
    final = trainer.run_evaluation()
    print(f"Ergebnis: Pruefung Ø {final:.2f}   "
          f"bester Pruefungsschnitt {trainer.eval_best:.2f}   "
          f"beste Einzelpartie {max(trainer.best_score, trainer.eval_max)}")
    print(f"Champion: {trainer.champion_path or '(noch keiner gespeichert)'}")
    print("Zuschauen mit:  python watch_ai.py dqn")


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN-Training fuer Snake")
    parser.add_argument("--headless", type=float, metavar="MINUTEN",
                        help="ohne Fenster trainieren, so viele Minuten lang")
    parser.add_argument("--weiter", action="store_true",
                        help="auf dem gespeicherten Champion aufbauen")
    args = parser.parse_args()

    if args.headless:
        run_headless(args.headless, args.weiter)
    else:
        from dashboard.dqn_view import main as run_dashboard
        run_dashboard(DQNConfig())


if __name__ == "__main__":
    main()
