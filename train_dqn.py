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
    python train_dqn.py --headless 480 --wahrnehmung rich_grid7
        (frischer Lauf mit einer ANDEREN Wahrnehmung als dem Code-Standard
         "rich" -- ohne das landet ein frischer Headless-Lauf immer bei
         "rich", weil Headless kein Menue hat)
    python train_dqn.py --headless 45 --seed 1 --curriculum 0.25
        (A/B-Werkzeuge: fester Seed + Curriculum-Anteil fuer diesen einen
         Lauf, ohne config.py anzufassen -- TRAININGSPLAN.md C-10)
    python train_dqn.py --headless 45 --pfadfokus 0.7
        (Pfad-Fokus-Regler: 0=reines Sammeln (Standard) .. 1=reines
         Ueberleben, Frucht bringt dann keine Belohnung mehr. Blendet
         automatisch aus, sobald ein sicherer Pfad gefunden ist)

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


def _apply_overrides(cfg: DQNConfig, overrides: list[str]) -> None:
    """Wendet generische `--override feld=wert`-Angaben auf die Config an.

    Gedacht fuer den Auto-Tuner (auto_tuner.py) und Power-Nutzung: JEDES
    DQNConfig-Feld laesst sich damit fuer einen einzelnen Headless-Lauf
    setzen, ohne config.py anzufassen. Der Wert wird als Python-Literal
    gelesen (Zahlen, True/False, Tupel wie "(256, 128)", Strings in
    Anfuehrungszeichen); was sich nicht parsen laesst, gilt als roher String.
    Unbekannte Feldnamen sind ein harter Fehler -- ein stiller Tippfehler
    wuerde sonst einen kompletten (teuren) Lauf mit falschen Annahmen
    produzieren.
    """
    import ast
    valid = set(vars(cfg).keys())
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--override erwartet FELD=WERT (bekommen: '{item}')")
        field, raw = item.split("=", 1)
        field = field.strip()
        if field not in valid:
            raise SystemExit(
                f"--override: unbekanntes Config-Feld '{field}'. "
                f"Moeglich: {', '.join(sorted(valid))}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        setattr(cfg, field, value)


def run_headless(minutes: float, resume: bool, brett: str | None = None,
                  wahrnehmung: str | None = None, seed: int | None = None,
                  curriculum: float | None = None,
                  pfadfokus: float | None = None,
                  overrides: list[str] | None = None) -> None:
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
    # Ohne --wahrnehmung wuerde ein frischer Headless-Lauf immer beim
    # Code-Standard ("rich") landen -- das Fenster-Menue erlaubt zwar eine
    # andere Wahl (z.B. "rich_grid7"), aber Headless hat kein Menue. Mit
    # dieser Option muss man dafuer nicht extra das Fenster oeffnen.
    if wahrnehmung:
        cfg.perception = wahrnehmung
    # A/B-Werkzeuge (TRAININGSPLAN.md C-10): fester Seed macht zwei Laeufe
    # vergleichbar; --curriculum/--pfadfokus schalten genau EINEN Unterschied
    # pro Vergleich, ohne dass man config.py anfassen muss.
    if seed is not None:
        cfg.seed = seed
    if curriculum is not None:
        cfg.curriculum_anteil = curriculum
    if pfadfokus is not None:
        cfg.pfad_fokus = pfadfokus
    # Generische Overrides ZULETZT anwenden -- sie sollen alles andere
    # (auch --wahrnehmung/--curriculum/...) uebersteuern koennen.
    if overrides:
        _apply_overrides(cfg, overrides)

    trainer = MultiGameTrainer(cfg, log_to_csv=True, resume_from=resume_path)

    print(f"Brett: {cfg.grid_cols}x{cfg.grid_rows}   "
          f"Wahrnehmung: {cfg.perception} ({trainer.input_size} Werte)   "
          f"Netz: {cfg.hidden}   Spiele: {cfg.num_games}   "
          f"Threads: {trainer.agent.threads}   Seed: {trainer.base_seed}")
    extras = []
    if cfg.curriculum_anteil:
        extras.append(f"Curriculum-Anteil {cfg.curriculum_anteil} "
                      f"({len(trainer.curriculum_snapshots)} Stellungen geladen)")
    if cfg.pfad_fokus:
        frucht_anteil = 1.0 - cfg.pfad_fokus
        extras.append(f"Pfad-Fokus {cfg.pfad_fokus:.0%} "
                      f"(Frucht zaehlt noch {frucht_anteil:.0%}, "
                      f"+{cfg.pfad_fokus_bonus * cfg.pfad_fokus:.3f}/Zug Bonus, "
                      f"blendet ab Lauf-Niveau {cfg.pfad_fokus_aus_ab:.0f} aus)")
    if extras:
        print("   ".join(extras))
    if trainer.eval_best > 0 and not resume_path:
        print(f"Champion-Schutzschwelle dieses Bretts: {trainer.eval_best:.1f} "
              "(die Champion-Datei wird nur ueberschrieben, wenn eine Pruefung "
              "darueber liegt -- der beste Stand DIESES Laufs landet unabhaengig "
              "davon in der Runbest-Datei).")
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
                    # Lauf-Bestwert anzeigen, NICHT die Champion-Schutzschwelle
                    # der Datei -- die stammt evtl. von einem ganz anderen Lauf
                    # und hat hier monatelang faelschlich "best 87.2" angezeigt.
                    print(f"{int(s.elapsed) // 60:3d}:{int(s.elapsed) % 60:02d}  "
                          f"Ep {s.total_episodes:6d}  "
                          f"Pruefung {pruefung}  (Lauf-best {s.eval_best_run:5.1f})  "
                          f"Training Ø {s.mean_score:5.1f}  "
                          f"eps {s.epsilon:.3f}  "
                          f"{rate:6.0f} Zuege/s", flush=True)
        except KeyboardInterrupt:
            print("\nAbgebrochen.")

        print("\nAbschluss-Pruefung laeuft ...")
        trainer.cfg.eval_episodes = 30
        final = trainer.run_evaluation()
        print(f"Ergebnis: Pruefung Ø {final:.2f}   "
              f"bester Pruefungsschnitt dieses Laufs {trainer.eval_best_run:.2f}   "
              f"beste Einzelpartie {max(trainer.best_score, trainer.eval_max)}")
        print(f"Champion: {trainer.champion_path or '(Schutzschwelle nicht uebertroffen)'}")
        print(f"Bester Stand dieses Laufs: {trainer.runbest_saved or '(keine Pruefung gelaufen)'}")
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
    parser.add_argument("--wahrnehmung", metavar="NAME",
                        help="Wahrnehmung fuer einen FRISCHEN Headless-Lauf, "
                             "z.B. rich_grid7 (siehe ai/perception.py fuer "
                             "alle Namen) -- ohne das landet ein frischer "
                             "Headless-Lauf immer beim Code-Standard 'rich'. "
                             "Nur im --headless-Modus; im Fenster gibt es "
                             "dafuer die Menue-Zeile 'Wahrnehmung'.")
    parser.add_argument("--seed", type=int, metavar="N",
                        help="fester Zufalls-Seed -- macht A/B-Laeufe "
                             "vergleichbar (TRAININGSPLAN.md C-10). Nur "
                             "--headless.")
    parser.add_argument("--curriculum", type=float, metavar="ANTEIL",
                        help="Endspiel-Curriculum-Anteil 0..1 fuer diesen "
                             "Lauf (0 = aus) -- fuer saubere A/B-Vergleiche "
                             "Curriculum an/aus. Nur --headless.")
    parser.add_argument("--pfadfokus", type=float, metavar="ANTEIL",
                        help="Pfad-Fokus-Regler 0.0..1.0: 0 = reines "
                             "Fruchtsammeln (Standard), 1 = reines "
                             "Ueberleben (Frucht bringt dann NULL Belohnung, "
                             "egal ob gefressen). Dazwischen linear gemischt. "
                             "Blendet automatisch Richtung 0 aus, sobald das "
                             "Lauf-Niveau steigt -- 'erst sicher, dann "
                             "schnell'. Nur --headless.")
    parser.add_argument("--override", action="append", metavar="FELD=WERT",
                        help="beliebiges DQNConfig-Feld fuer diesen Lauf "
                             "setzen, z.B. --override gamma=0.99 --override "
                             "'hidden=(256, 256)'. Mehrfach angebbar; wird "
                             "NACH allen anderen Flags angewandt. Haupt-"
                             "verwendung: auto_tuner.py. Nur --headless.")
    args = parser.parse_args()

    if args.headless:
        run_headless(args.headless, args.weiter, args.brett, args.wahrnehmung,
                     args.seed, args.curriculum, args.pfadfokus, args.override)
    elif args.brett:
        raise SystemExit("--brett wird nur mit --headless unterstuetzt -- "
                          "im Fenster die Menue-Zeile 'Brettgroesse' benutzen.")
    elif args.wahrnehmung:
        raise SystemExit("--wahrnehmung wird nur mit --headless unterstuetzt -- "
                          "im Fenster die Menue-Zeile 'Wahrnehmung' benutzen.")
    elif (args.seed is not None or args.curriculum is not None
          or args.pfadfokus is not None or args.override):
        raise SystemExit("--seed/--curriculum/--pfadfokus/--override werden "
                          "nur mit --headless unterstuetzt.")
    else:
        from dashboard.dqn_view import main as run_dashboard
        run_dashboard(resume=args.weiter)


if __name__ == "__main__":
    main()
