"""Autonomer Konfigurations-Tuner: sucht selbststaendig die beste DQN-Config.

    source venv/bin/activate
    python auto_tuner.py                          # laeuft bis Konvergenz/24h
    python auto_tuner.py --minuten 15             # 15 Min je Trainings-Lauf
    python auto_tuner.py --max-stunden 8          # abends starten, morgens lesen
    python auto_tuner.py --fortsetzen logs/autotuner-XXXX/zustand.json

Was er tut (Lucas Auftrag vom 2026-07-24):
1. Startet mit einer Basis-Konfiguration und misst sie ehrlich (mehrere
   Seeds, gleiches Zeitbudget je Lauf).
2. Aendert dann GENAU EINE Einstellung (TRAININGSPLAN.md C-10: nie mehr als
   einen Unterschied pro Vergleich), trainiert damit, vergleicht:
   besser -> die Aenderung wird behalten; gleich oder schlechter -> zurueck
   zur vorigen Einstellung.
3. Verworfene Werte kommen mit einer festen Wahrscheinlichkeit spaeter
   ERNEUT auf den Pruefstand (bis zu 3x) -- so wird ausgeschlossen, dass
   ein einzelner Pech-Lauf einen eigentlich guten Wert beerdigt hat.
4. Alles wird protokolliert: jeder einzelne Lauf (Config, Seeds, Scores,
   Report-Pfad), jede Entscheidung, der komplette Verlauf -- als CSV +
   Markdown + maschinenlesbarer Zustand. Der Abschlussbericht fasst am Ende
   zusammen, welche Werte sich wie geschlagen haben.
5. Stoppt von selbst, wenn lange nichts mehr besser wird (Konvergenz),
   nach --max-stunden, oder sauber per Strg+C (schreibt auch dann den
   Abschlussbericht; der laufende Trainings-Prozess schreibt dank seines
   finally-Blocks ebenfalls noch seinen eigenen Report).

Warum EIN Lauf nach dem anderen statt 5 parallel (bewusste Entscheidung):
Die Bewertung ist "was schafft der Bot in X Minuten WANDUHR-Zeit". Fuenf
gleichzeitige Trainings wuerden sich CPU-Kerne und Waermebudget teilen --
jeder Lauf waere unterschiedlich stark ausgebremst, je nachdem was zufaellig
parallel laeuft. Die Scores waeren damit unvergleichbar und die ganze Suche
wertlos. Dazu kaemen Datei-Kollisionen (Champion/Runbest/Stellungen sind
EINE Datei pro Brett). Sequenziell ist bei diesem Messprinzip die einzige
ehrliche Variante -- dafuer laeuft sie zuverlaessig tagelang durch.

Leitplanke: Der Tuner fasst NUR Trainings-Stellschrauben an (DQNConfig).
Er aendert nie Spielregeln und gibt der KI keinerlei Strategie -- er dreht
an denselben Reglern, die auch im Menue stehen, nur eben automatisch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_ROOT, "logs")
MODEL_DIR = os.path.join(_ROOT, "models")

# ------------------------------------------------------------------------- #
# SUCHRAUM: jedes Feld mit seinen Kandidaten-Werten. Der Tuner probiert
# immer nur Werte aus diesen Listen -- bewusst endliche, sinnvolle Stufen
# statt beliebiger Zahlen (jede Stufe ist einzeln interpretierbar, und der
# Abschlussbericht kann "Wert X war im Schnitt so gut" sauber auswerten).
# ------------------------------------------------------------------------- #
SUCHRAUM: dict[str, list] = {
    "learning_rate": [5e-4, 1e-3, 2e-3],
    "gamma": [0.95, 0.97, 0.99],
    "n_step": [1, 3, 5],
    "batch_size": [128, 256, 512],
    "hidden": [(128, 128), (256, 128), (256, 256), (512, 256)],
    "num_games": [8, 16, 24],
    "fruit_count": [1, 3, 5, 10],
    "eps_decay_steps": [40_000, 80_000, 150_000],
    "eps_end": [0.02, 0.05],
    "train_every": [1, 2],
    "target_update": [500, 1_000, 2_000],
    "perception": ["rich", "rich_grid5", "rich_grid7", "rich_grid9"],
    "curriculum_anteil": [0.0, 0.25, 0.5, 1.0],
    "pfad_fokus": [0.0, 0.25, 0.5, 0.7, 1.0],
    "pfad_fokus_bonus": [0.02, 0.05, 0.1],
    "balance_anteil": [0.0, 0.3, 0.5],
    "spiegel_lernen": [True, False],
    "prioritized": [False, True],
    "activation": ["relu", "tanh"],
    "reward_death": [-10.0, -20.0],
    "reward_step": [-0.01, 0.0],
}

# Startpunkt der Suche: Code-Standard + die Menue-Empfehlung rich_grid7.
BASIS_ABWEICHUNGEN: dict[str, object] = {
    "perception": "rich_grid7",
}

# Feste Seed-Liste -- jeder Kandidat wird mit DENSELBEN Seeds gemessen wie
# die aktuelle Bestmarke, sonst vergleicht man Aepfel mit Birnen.
SEED_POOL = [11, 23, 37, 51]

RETEST_WAHRSCHEINLICHKEIT = 0.2   # Anteil der Schritte, die einen verworfenen Wert erneut pruefen
MAX_RETESTS = 3                   # so oft darf ein verworfener Wert insgesamt erneut antreten
REBASE_ALLE = 8                   # alle N Kandidaten die Bestmarke frisch nachmessen
                                  # (gegen Waerme-Drift: der Mac wird im Tagesverlauf langsamer)


# ------------------------------------------------------------------------- #
# Einen einzelnen Trainings-Lauf ausfuehren und bewerten
# ------------------------------------------------------------------------- #
def run_training(overrides: dict, seed: int, minuten: float) -> dict:
    """Startet EINEN Headless-Trainingslauf als Unterprozess und liest
    dessen Report. Rueckgabe: {score, report_json, run_id, zuege_pro_s, ok}.
    score = bester Pruefungsschnitt des Laufs (inkl. Abschluss-Pruefung)."""
    cmd = [sys.executable, "train_dqn.py", "--headless", str(minuten),
           "--seed", str(seed)]
    for feld, wert in overrides.items():
        cmd += ["--override", f"{feld}={wert!r}"]

    try:
        proc = subprocess.run(
            cmd, cwd=_ROOT, capture_output=True, text=True,
            timeout=minuten * 60 + 600,   # grosszuegiger Puffer fuer Abschluss-Pruefung
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "fehler": "Zeitueberschreitung", "score": None}

    match = re.search(r"Report: (.*-report\.md)", proc.stdout)
    if proc.returncode != 0 or not match:
        return {"ok": False, "score": None,
                "fehler": (proc.stderr or proc.stdout)[-500:]}

    json_path = match.group(1).replace("-report.md", "-report.json")
    try:
        with open(json_path, encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "score": None, "fehler": f"Report unlesbar: {exc}"}

    return {
        "ok": True,
        "score": float(report["pruefung"]["bestwert_dieses_laufs"]),
        "report_json": json_path,
        "run_id": os.path.basename(json_path).replace("-report.json", ""),
        "zuege_pro_s": report.get("zuege_pro_sekunde"),
    }


def messe_config(overrides: dict, seeds: list[int], minuten: float,
                 protokoll) -> tuple[float | None, list[dict]]:
    """Misst eine Config ueber mehrere Seeds; Score = Mittel der Laeufe.
    None, wenn ein Lauf fehlschlaegt (dann ist der Kandidat ungueltig)."""
    laeufe = []
    for seed in seeds:
        ergebnis = run_training(overrides, seed, minuten)
        ergebnis["seed"] = seed
        laeufe.append(ergebnis)
        protokoll.lauf(overrides, ergebnis)
        if not ergebnis["ok"]:
            return None, laeufe
    scores = [l["score"] for l in laeufe]
    return sum(scores) / len(scores), laeufe


# ------------------------------------------------------------------------- #
# Protokoll: CSV + Markdown + Zustand, alles im eigenen Tuner-Ordner
# ------------------------------------------------------------------------- #
class Protokoll:
    def __init__(self, ordner: str) -> None:
        self.ordner = ordner
        os.makedirs(ordner, exist_ok=True)
        self.csv_path = os.path.join(ordner, "laeufe.csv")
        self.md_path = os.path.join(ordner, "protokoll.md")
        self.zustand_path = os.path.join(ordner, "zustand.json")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(
                    ["zeit", "seed", "score", "zuege_pro_s", "run_id", "config"])
            self._md("# Auto-Tuner Protokoll\n\nGestartet: "
                     + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    def _md(self, text: str) -> None:
        with open(self.md_path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    def lauf(self, overrides: dict, ergebnis: dict) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                time.strftime("%H:%M:%S"), ergebnis.get("seed"),
                ergebnis.get("score"), ergebnis.get("zuege_pro_s"),
                ergebnis.get("run_id", "FEHLGESCHLAGEN"),
                json.dumps(overrides, default=str, ensure_ascii=False),
            ])

    def entscheidung(self, text: str) -> None:
        print(text, flush=True)
        self._md(f"- {time.strftime('%H:%M')} {text}")

    def zustand(self, daten: dict) -> None:
        with open(self.zustand_path, "w", encoding="utf-8") as fh:
            json.dump(daten, fh, indent=2, default=str, ensure_ascii=False)


def schreibe_abschlussbericht(ordner: str, zustand: dict) -> str:
    """Fasst am Ende zusammen: beste Config, kompletter Verlauf, und je
    Parameter eine "was haben wir gelernt"-Tabelle (Durchschnitts-Score
    aller Messungen je Wert)."""
    pfad = os.path.join(ordner, "abschlussbericht.md")
    zeilen = [
        "# Auto-Tuner Abschlussbericht",
        f"Erzeugt: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Kandidaten getestet: {zustand['kandidaten_gesamt']}  ·  "
        f"davon angenommen: {zustand['verbesserungen']}  ·  "
        f"Trainings-Laeufe insgesamt: {zustand['laeufe_gesamt']}",
        "",
        f"## Beste gefundene Konfiguration (Score {zustand['best_score']:.2f})",
        "```",
        json.dumps(zustand["best_overrides"], indent=2, default=str,
                   ensure_ascii=False),
        "```",
        "(nur Abweichungen vom Code-Standard; als Startbefehl: "
        "`python train_dqn.py --headless 480` + je Zeile ein "
        "`--override feld=wert`)",
        "", "## Verlauf der Bestmarke",
    ]
    for eintrag in zustand["verlauf"]:
        zeilen.append(f"- {eintrag}")

    # Was haben wir je Parameter gelernt? Durchschnitt aller Messungen je Wert.
    zeilen += ["", "## Gelerntes je Parameter (Ø Score aller Messungen mit diesem Wert)"]
    pro_wert: dict[str, dict[str, list[float]]] = {}
    for kandidat in zustand["historie"]:
        if kandidat["mittel"] is None:
            continue
        wert_key = str(kandidat["neuer_wert"])
        pro_wert.setdefault(kandidat["parameter"], {}).setdefault(
            wert_key, []).append(kandidat["mittel"])
    for param in sorted(pro_wert):
        zeilen.append(f"\n**{param}**")
        werte = pro_wert[param]
        for wert, scores in sorted(werte.items(),
                                   key=lambda kv: -sum(kv[1]) / len(kv[1])):
            mittel = sum(scores) / len(scores)
            zeilen.append(f"- {wert}: Ø {mittel:.2f}  ({len(scores)} Messung(en))")

    with open(pfad, "w", encoding="utf-8") as fh:
        fh.write("\n".join(zeilen) + "\n")
    return pfad


# ------------------------------------------------------------------------- #
# Hauptschleife
# ------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomer DQN-Config-Tuner")
    parser.add_argument("--minuten", type=float, default=12.0,
                        help="Trainingszeit je Lauf (Standard 12)")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Seeds je Messung (Standard 2 -- mehr = ehrlicher, langsamer)")
    parser.add_argument("--max-stunden", type=float, default=24.0,
                        help="spaetestens nach so vielen Stunden aufhoeren")
    parser.add_argument("--konvergenz", type=int, default=30,
                        help="aufhoeren, wenn so viele Kandidaten in Folge verworfen wurden")
    parser.add_argument("--marge", type=float, default=0.02,
                        help="Mindest-Verbesserung (relativ), um eine Aenderung zu behalten")
    parser.add_argument("--max-kandidaten", type=int, default=0,
                        help="0 = unbegrenzt (nur fuer Tests gedacht)")
    parser.add_argument("--fortsetzen", metavar="ZUSTAND_JSON",
                        help="einen frueheren Tuner-Lauf fortsetzen")
    args = parser.parse_args()

    seeds = SEED_POOL[:max(1, min(args.seeds, len(SEED_POOL)))]
    rng = random.Random()

    if args.fortsetzen:
        with open(args.fortsetzen, encoding="utf-8") as fh:
            zustand = json.load(fh)
        # hidden kommt aus JSON als Liste zurueck -- fuer DQNConfig als Tupel.
        if "hidden" in zustand["best_overrides"]:
            zustand["best_overrides"]["hidden"] = tuple(zustand["best_overrides"]["hidden"])
        ordner = os.path.dirname(os.path.abspath(args.fortsetzen))
        protokoll = Protokoll(ordner)
        protokoll.entscheidung(f"FORTSETZUNG: Bestmarke {zustand['best_score']:.2f}")
    else:
        ordner = os.path.join(LOG_DIR, f"autotuner-{time.strftime('%Y%m%d-%H%M%S')}")
        protokoll = Protokoll(ordner)

        # Sicherung der Modell-Dateien des Bretts -- der Tuner ueberschreibt
        # models/dqn_runbest_*.pt bei jedem Lauf; der Champion ist durch
        # seine Schutzschwelle sicher, aber doppelt haelt besser.
        sicherung = os.path.join(ordner, "modell-sicherung")
        os.makedirs(sicherung, exist_ok=True)
        for name in os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []:
            quelle = os.path.join(MODEL_DIR, name)
            if os.path.isfile(quelle):
                shutil.copy2(quelle, os.path.join(sicherung, name))
        protokoll.entscheidung(f"Modell-Sicherung nach {sicherung}")

        protokoll.entscheidung(
            f"BASIS messen: {BASIS_ABWEICHUNGEN} mit Seeds {seeds}, "
            f"{args.minuten:g} Min je Lauf")
        basis_score, _ = messe_config(dict(BASIS_ABWEICHUNGEN), seeds,
                                      args.minuten, protokoll)
        if basis_score is None:
            raise SystemExit("Basis-Messung fehlgeschlagen -- siehe laeufe.csv")
        protokoll.entscheidung(f"BASIS: Score {basis_score:.2f}")

        zustand = {
            "best_overrides": dict(BASIS_ABWEICHUNGEN),
            "best_score": basis_score,
            "historie": [],          # ein Eintrag je Kandidat
            "verlauf": [f"Start: Basis {basis_score:.2f} ({BASIS_ABWEICHUNGEN})"],
            "verworfen": [],         # Kandidaten fuer spaetere Retests
            "kandidaten_gesamt": 0,
            "verbesserungen": 0,
            "laeufe_gesamt": len(seeds),
            "in_folge_verworfen": 0,
            "seeds": seeds,
            "minuten": args.minuten,
        }
        protokoll.zustand(zustand)

    start = time.time()
    try:
        while True:
            if args.max_kandidaten and zustand["kandidaten_gesamt"] >= args.max_kandidaten:
                protokoll.entscheidung("STOP: --max-kandidaten erreicht.")
                break
            if (time.time() - start) / 3600 >= args.max_stunden:
                protokoll.entscheidung("STOP: --max-stunden erreicht.")
                break
            if zustand["in_folge_verworfen"] >= args.konvergenz:
                protokoll.entscheidung(
                    f"KONVERGENZ: {args.konvergenz} Kandidaten in Folge "
                    "verworfen -- die Suche gilt als ausgereizt.")
                break

            # Alle REBASE_ALLE Kandidaten: Bestmarke frisch nachmessen, damit
            # Waerme-Drift (der Mac wird ueber Stunden langsamer) nicht die
            # alte Bestmarke kuenstlich unschlagbar macht.
            if (zustand["kandidaten_gesamt"] > 0
                    and zustand["kandidaten_gesamt"] % REBASE_ALLE == 0
                    and zustand.get("_letzte_rebase") != zustand["kandidaten_gesamt"]):
                zustand["_letzte_rebase"] = zustand["kandidaten_gesamt"]
                neu, _ = messe_config(zustand["best_overrides"], seeds,
                                      args.minuten, protokoll)
                if neu is not None:
                    protokoll.entscheidung(
                        f"NACHMESSUNG Bestmarke: {zustand['best_score']:.2f} -> {neu:.2f}")
                    zustand["best_score"] = neu
                    zustand["laeufe_gesamt"] += len(seeds)
                    protokoll.zustand(zustand)

            # ---- Kandidat waehlen ---------------------------------------- #
            retest_pool = [v for v in zustand["verworfen"] if v["retests"] < MAX_RETESTS]
            if retest_pool and rng.random() < RETEST_WAHRSCHEINLICHKEIT:
                kandidat = rng.choice(retest_pool)
                kandidat["retests"] += 1
                param, wert = kandidat["parameter"], kandidat["neuer_wert"]
                if isinstance(wert, list):
                    wert = tuple(wert)   # hidden kommt aus JSON als Liste
                grund = f"RETEST #{kandidat['retests']}"
            else:
                param = rng.choice(list(SUCHRAUM.keys()))
                aktuell = zustand["best_overrides"].get(param)
                from ai.dqn.config import DQNConfig
                if aktuell is None:
                    aktuell = getattr(DQNConfig(), param)
                # Tupel-Werte (hidden) koennen nach einem JSON-Roundtrip als
                # Liste vorliegen -- fuer den Vergleich beides angleichen.
                if isinstance(aktuell, (list, tuple)):
                    optionen = [w for w in SUCHRAUM[param]
                                if tuple(w) != tuple(aktuell)]
                else:
                    optionen = [w for w in SUCHRAUM[param] if w != aktuell]
                if not optionen:
                    continue
                wert = rng.choice(optionen)
                grund = "NEU"

            kandidat_overrides = dict(zustand["best_overrides"])
            kandidat_overrides[param] = wert

            zustand["kandidaten_gesamt"] += 1
            nr = zustand["kandidaten_gesamt"]
            protokoll.entscheidung(
                f"Kandidat #{nr} ({grund}): {param} -> {wert!r} "
                f"(Bestmarke {zustand['best_score']:.2f})")

            mittel, laeufe = messe_config(kandidat_overrides, seeds,
                                          args.minuten, protokoll)
            zustand["laeufe_gesamt"] += len(laeufe)

            eintrag = {
                "nr": nr, "grund": grund, "parameter": param,
                "neuer_wert": wert, "mittel": mittel,
                "scores": [l.get("score") for l in laeufe],
                "run_ids": [l.get("run_id") for l in laeufe],
                "bestmarke_davor": zustand["best_score"],
            }
            zustand["historie"].append(eintrag)

            # ---- Entscheidung ------------------------------------------- #
            if mittel is not None and mittel > zustand["best_score"] * (1 + args.marge):
                zustand["best_overrides"] = kandidat_overrides
                alt = zustand["best_score"]
                zustand["best_score"] = mittel
                zustand["verbesserungen"] += 1
                zustand["in_folge_verworfen"] = 0
                zustand["verworfen"] = [v for v in zustand["verworfen"]
                                        if v["parameter"] != param]
                zustand["verlauf"].append(
                    f"#{nr}: {param}={wert!r} ANGENOMMEN "
                    f"({alt:.2f} -> {mittel:.2f})")
                protokoll.entscheidung(
                    f"  => ANGENOMMEN: {mittel:.2f} schlaegt {alt:.2f}")
            else:
                zustand["in_folge_verworfen"] += 1
                anzeige = "FEHLGESCHLAGEN" if mittel is None else f"{mittel:.2f}"
                protokoll.entscheidung(
                    f"  => verworfen ({anzeige} vs. {zustand['best_score']:.2f})")
                if grund == "NEU" and mittel is not None:
                    zustand["verworfen"].append({
                        "parameter": param, "neuer_wert": wert,
                        "retests": 0, "letztes_mittel": mittel,
                    })

            protokoll.zustand(zustand)

    except KeyboardInterrupt:
        protokoll.entscheidung("STOP: per Strg+C beendet.")

    pfad = schreibe_abschlussbericht(ordner, zustand)
    protokoll.zustand(zustand)
    print(f"\nAbschlussbericht: {pfad}")
    print(f"Beste Config (Score {zustand['best_score']:.2f}): "
          f"{json.dumps(zustand['best_overrides'], default=str, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
