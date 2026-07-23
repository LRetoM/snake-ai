"""Startbefehl fuer das Deep-Q-Learning-Training (die ZWEITE, eigenstaendige KI).

Start (mit Fenster, zum Zusehen):
    source venv/bin/activate      (Mac)   /   venv\\Scripts\\activate   (Windows)
    python train_dqn.py

Start OHNE Fenster (maximaler Lern-Durchsatz, z.B. ueber Nacht):
    python train_dqn.py --headless 30      (30 Minuten rechnen)
    python train_dqn.py --headless 30 --weiter   (auf dem Champion aufbauen)
    python train_dqn.py --headless 480 --weiter --brett 17x15
        (Brett-Transfer: baut auf dem Champion des Standard-Bretts auf,
         trainiert aber auf dem angegebenen Brett weiter -- siehe
         TRAININGSPLAN.md Abschnitt B3/S0.4 fuers Klein->Gross-Curriculum)

Der Unterschied: ohne Fenster faellt das Zeichnen komplett weg -- je nach
Rechner sind das nochmal 10-30% mehr Zuege pro Sekunde als der Turbo-Modus im
Fenster. Der Fortschritt wird stattdessen alle 30 Sekunden in die Konsole
geschrieben, und der beste Bot landet in models/dqn_champion_<cols>x<rows>.pt
(jede Brettgroesse hat ihre eigene Champion-Datei).

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


def _parse_board(text: str) -> tuple[int, int]:
    """Wandelt "17x15" (oder "17X15") in (17, 15) um."""
    try:
        cols_s, rows_s = text.lower().split("x")
        return int(cols_s), int(rows_s)
    except ValueError as exc:
        raise SystemExit(
            f"--brett erwartet das Format SPALTENxZEILEN, z.B. 17x15 (bekommen: '{text}')"
        ) from exc


def _resolve_resume_path(target_cols: int, target_rows: int):
    """Welche Champion-Datei fuer --weiter (+optional --brett) geladen wird.

    1) Gibt es fuer das ZIEL-Brett schon einen Champion -> ganz normales
       Weitertrainieren auf demselben Brett.
    2) Sonst (Brett-Transfer, z.B. Klein-Feld-Curriculum): der Champion des
       Standard-Bretts (DQNConfig()-Default) dient als Ausgangspunkt --
       seine Gewichte werden geladen, das Brett aber auf das Ziel gesetzt.
       Fuer mehrstufige Curricula (9x9 -> 13x11 -> 17x15) empfiehlt sich
       stattdessen das Fenster-Menue (siehe TRAININGSPLAN.md Abschnitt B3),
       das gezielt den Champion EINES bestimmten Zwischenbretts laedt.
    """
    from ai.dqn.trainer import resolve_champion_path

    path = resolve_champion_path(target_cols, target_rows)
    if path:
        return path
    default_cfg = DQNConfig()
    if (default_cfg.grid_cols, default_cfg.grid_rows) != (target_cols, target_rows):
        return resolve_champion_path(default_cfg.grid_cols, default_cfg.grid_rows)
    return None


def run_headless(minutes: float, resume: bool, brett: str | None = None) -> None:
    """Trainiert ohne jede Grafik und meldet den Fortschritt in der Konsole."""
    from ai.dqn.trainer import MultiGameTrainer, load_champion_config

    base_cfg = DQNConfig()
    target_cols, target_rows = (
        _parse_board(brett) if brett else (base_cfg.grid_cols, base_cfg.grid_rows)
    )

    resume_path = _resolve_resume_path(target_cols, target_rows) if resume else None
    # Weitertrainieren heisst: auch die Einstellungen exakt so uebernehmen, mit
    # denen dieser Champion gezuechtet wurde -- sonst wuerden Netzgroesse,
    # Lernrate, Fruechte etc. wieder auf die Code-Standardwerte zurueckfallen,
    # obwohl der Champion vielleicht ganz anders (und besser) eingestellt war.
    cfg = load_champion_config(resume_path) if resume_path else None
    if cfg is None:
        cfg = DQNConfig()
    board_transfer = (cfg.grid_cols, cfg.grid_rows) != (target_cols, target_rows)
    cfg.grid_cols, cfg.grid_rows = target_cols, target_rows

    trainer = MultiGameTrainer(cfg, log_to_csv=True, resume_from=resume_path)

    print(f"Brett: {cfg.grid_cols}x{cfg.grid_rows}   "
          f"Wahrnehmung: {cfg.perception} ({trainer.input_size} Werte)   "
          f"Netz: {cfg.hidden}   Spiele: {cfg.num_games}   "
          f"Threads: {trainer.agent.threads}")
    if resume_path and board_transfer:
        print(f"Brett-Transfer: Gewichte von {resume_path} uebernommen, "
              f"trainiert jetzt auf {cfg.grid_cols}x{cfg.grid_rows} weiter "
              "(Rekorde starten neu -- Scores sind zwischen Brettgroessen "
              "nicht vergleichbar).")
    elif resume_path:
        print(f"Weitertrainiert auf: {resume_path}  (Einstellungen vom Champion uebernommen)")
    print(f"Laeuft {minutes:g} Minuten. Abbrechen mit Strg+C.\n")

    t_end = time.perf_counter() + minutes * 60
    next_console_print = time.perf_counter() + 30
    last_moves, last_t = 0, time.perf_counter()
    # Der Report (TRAININGSPLAN.md 0.1) wird IMMER geschrieben, egal wie der
    # Lauf endet -- normal durchgelaufen, Strg+C, oder ein Fehler dazwischen.
    # Sonst waere ausgerechnet ein abgebrochener Lauf der eine, ueber den man
    # hinterher nichts mehr auswerten kann.
    try:
        try:
            while time.perf_counter() < t_end:
                for _ in range(500):
                    trainer.step()
                now = time.perf_counter()
                if now >= next_console_print:
                    next_console_print = now + 30
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
    finally:
        _json_path, md_path = trainer.write_report()
        print(f"Report: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN-Training fuer Snake")
    parser.add_argument("--headless", type=float, metavar="MINUTEN",
                        help="ohne Fenster trainieren, so viele Minuten lang")
    parser.add_argument("--weiter", action="store_true",
                        help="auf dem gespeicherten Champion aufbauen")
    parser.add_argument("--brett", metavar="SPALTENxZEILEN",
                        help="Brettgroesse fuer diesen Lauf, z.B. 17x15 -- "
                             "nur zusammen mit --weiter sinnvoll (Brett-"
                             "Transfer, siehe TRAININGSPLAN.md). Nur im "
                             "--headless-Modus; im Fenster gibt es dafuer "
                             "die Menue-Zeile 'Brettgroesse'.")
    args = parser.parse_args()

    if args.headless:
        run_headless(args.headless, args.weiter, args.brett)
    elif args.brett:
        raise SystemExit("--brett wird nur mit --headless unterstuetzt -- "
                          "im Fenster die Menue-Zeile 'Brettgroesse' benutzen.")
    else:
        from dashboard.dqn_view import main as run_dashboard
        run_dashboard(resume=args.weiter)


if __name__ == "__main__":
    main()
