"""Autonomer Konfigurations-Tuner: sucht selbststaendig die beste DQN-Config.

    source venv/bin/activate
    python auto_tuner.py                       # laeuft bis Konvergenz (Standard)
    python auto_tuner.py --minuten 10          # 10 Min je Trainings-Lauf
    python auto_tuner.py --fortsetzen logs/autotuner-XXX/zustand.json

Wie er sucht -- "Coordinate Ascent" (Lucas Auftrag vom 2026-07-24):
Statt zufaellig herumzuprobieren, geht der Tuner in DURCHLAEUFEN systematisch
vor. Ein Durchlauf testet JEDEN Parameter der Reihe nach: fuer den aktuellen
Parameter werden ALLE moeglichen Werte gemessen, der beste uebernommen, dann
zum naechsten Parameter. So ist garantiert, dass "erst alles einmal
ausprobiert" wurde, bevor irgendeine Entscheidung ueber Konvergenz faellt.

Konvergenz ist dadurch KLAR definiert: sobald ein KOMPLETTER Durchlauf ueber
alle Parameter keine einzige Verbesserung mehr bringt, gilt das Optimum als
bestaetigt -- der Tuner schaltet sich dann SELBST ab und schreibt den
Abschlussbericht. (Kein festes 24h-Limit; `--max-stunden` ist nur eine
Sicherheits-Obergrenze.)

Retests sind eingebaut, ohne Extra-Laeufe: jeder Durchlauf misst ALLE Werte
erneut gegen die (frisch nachgemessene) Bestmarke -- und mit einem ANDEREN
Seed als der vorige Durchlauf. Ein Wert, der nur durch einen Pech-/Glueck-
Lauf angenommen oder verworfen wurde, wird so im naechsten Durchlauf
zwangslaeufig widerlegt. Ein Wert, der ueber mehrere Durchlaeufe (= mehrere
Seeds) vorne bleibt, ist damit "zweifelsfrei" gut.

Ehrlichkeit gegenueber Waerme-Drift: der Mac wird ueber Stunden langsamer.
Deshalb wird zu BEGINN JEDES Durchlaufs die aktuelle Bestmarke frisch
nachgemessen -- alle Vergleiche INNERHALB eines Durchlaufs laufen gegen
diesen frischen Anker und mit demselben Seed, sind also fair.

Sicheres Beenden: Strg+C (oder Terminal schliessen) stoppt SAUBER -- der
gerade laufende Trainings-Lauf schreibt dank seines eigenen finally-Blocks
noch seinen Report, danach schreibt der Tuner Zustand + Abschlussbericht.
Nichts geht verloren. Fortsetzen jederzeit mit --fortsetzen.

Warum EIN Lauf nach dem anderen (nicht 5 parallel): die Bewertung ist "was
schafft der Bot in X Minuten WANDUHR-Zeit". Parallele Laeufe teilen sich
CPU + Waermebudget -- jeder waere unterschiedlich stark gebremst, die Scores
damit unvergleichbar und die Suche wertlos. Dazu Datei-Kollisionen (Champion/
Runbest/Stellungen sind EINE Datei pro Brett). Sequenziell ist die einzige
ehrliche Variante.

Leitplanke: Der Tuner dreht NUR an Trainings-Reglern (DQNConfig) -- dieselben,
die auch im Menue stehen. Er aendert nie Spielregeln und gibt der KI keine
Strategie.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time

from ai.dqn.config import DQNConfig

_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_ROOT, "logs")
MODEL_DIR = os.path.join(_ROOT, "models")

# ------------------------------------------------------------------------- #
# SUCHRAUM: jedes Feld mit seinen Kandidaten-Werten, jeweils AUFSTEIGEND
# sortiert (wichtig fuer die "Wert am Rand"-Empfehlung im Abschlussbericht).
# Bewusst endliche, sinnvolle Stufen statt beliebiger Zahlen -- jede Stufe
# ist einzeln interpretierbar.
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
    "spiegel_lernen": [False, True],
    "prioritized": [False, True],
    "activation": ["relu", "tanh"],
    "reward_death": [-20.0, -10.0],
    "reward_step": [-0.01, 0.0],
}

# Startpunkt der Suche: Code-Standard + die Menue-Empfehlung rich_grid7.
BASIS_ABWEICHUNGEN: dict[str, object] = {
    "perception": "rich_grid7",
}

# Seeds fuer die Messungen. Pro Durchlauf wird EIN anderer Seed(-block) aus
# diesem Pool benutzt (rotierend) -- so bekommt jeder Wert ueber die
# Durchlaeufe hinweg verschiedene Seeds zu sehen (eingebauter Retest).
SEED_POOL = [11, 23, 37, 51, 67, 83]


def _signatur(overrides: dict) -> str:
    """Eindeutiger Schluessel fuer eine Config (fuer den Mess-Cache). Tupel
    werden zu Listen normiert, Schluessel sortiert -- damit gleiche Configs
    garantiert denselben Schluessel ergeben."""
    norm = {k: (list(v) if isinstance(v, tuple) else v) for k, v in overrides.items()}
    return json.dumps(norm, sort_keys=True, default=str)


def _alternativen(param: str, best: dict) -> list:
    """Alle Werte von `param`, die vom aktuell besten abweichen."""
    aktuell = best.get(param)
    if aktuell is None:
        aktuell = getattr(DQNConfig(), param)
    out = []
    for w in SUCHRAUM[param]:
        if isinstance(aktuell, (list, tuple)):
            gleich = tuple(w) == tuple(aktuell)
        else:
            gleich = w == aktuell
        if not gleich:
            out.append(w)
    return out


# ------------------------------------------------------------------------- #
# Einen einzelnen Trainings-Lauf ausfuehren und bewerten
# ------------------------------------------------------------------------- #
def run_training(overrides: dict, seed: int, minuten: float) -> dict:
    """Startet EINEN Headless-Trainingslauf als Unterprozess und liest dessen
    Report. Rueckgabe: {ok, score, report_json, run_id, zuege_pro_s}.
    score = bester Pruefungsschnitt des Laufs (inkl. Abschluss-Pruefung)."""
    cmd = [sys.executable, "train_dqn.py", "--headless", str(minuten),
           "--seed", str(seed)]
    for feld, wert in overrides.items():
        wert_repr = repr(tuple(wert) if isinstance(wert, list) else wert)
        cmd += ["--override", f"{feld}={wert_repr}"]

    try:
        proc = subprocess.run(
            cmd, cwd=_ROOT, capture_output=True, text=True,
            timeout=minuten * 60 + 600,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "fehler": "Zeitueberschreitung", "score": None}
    except KeyboardInterrupt:
        # Strg+C erreicht auch den Kindprozess (Prozessgruppe) -- der schreibt
        # dank finally noch seinen Report. Hier nur sauber weiterreichen.
        raise

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
                    ["zeit", "durchlauf", "seed", "score", "zuege_pro_s",
                     "run_id", "config"])
            self._md("# Auto-Tuner Protokoll\n\nGestartet: "
                     + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    def _md(self, text: str) -> None:
        with open(self.md_path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    def lauf(self, durchlauf: int, overrides: dict, ergebnis: dict) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([
                time.strftime("%H:%M:%S"), durchlauf, ergebnis.get("seed"),
                ergebnis.get("score"), ergebnis.get("zuege_pro_s"),
                ergebnis.get("run_id", "FEHLGESCHLAGEN"),
                json.dumps(overrides, default=str, ensure_ascii=False),
            ])

    def notiz(self, text: str) -> None:
        print(text, flush=True)
        self._md(f"- {time.strftime('%H:%M')} {text}")

    def zustand(self, daten: dict) -> None:
        # Erst in eine Temp-Datei, dann umbenennen -- so bleibt zustand.json
        # auch dann heil, wenn genau beim Schreiben Strg+C kommt.
        tmp = self.zustand_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(daten, fh, indent=2, default=str, ensure_ascii=False)
        os.replace(tmp, self.zustand_path)


# ------------------------------------------------------------------------- #
# Sauberes Beenden: ein Signal-Handler setzt nur ein Flag. Der laufende
# Trainings-Unterprozess bekommt das Signal ueber die Prozessgruppe ohnehin
# selbst ab und schreibt seinen Report -- wir brechen danach kontrolliert ab.
# ------------------------------------------------------------------------- #
_STOP = {"flag": False}


def _handle_stop(signum, frame):
    if not _STOP["flag"]:
        print("\n[Tuner] Stopp angefordert -- beende nach dem aktuellen Lauf "
              "und schreibe den Abschlussbericht ...", flush=True)
    _STOP["flag"] = True


# ------------------------------------------------------------------------- #
# Abschlussbericht: die vollstaendige Auswertung
# ------------------------------------------------------------------------- #
def _empfehlungen(zustand: dict) -> list[str]:
    """Automatische Vorschlaege fuer weitere Verbesserungen.

    Nur fuer ZAHLEN-Parameter, die der Tuner AKTIV vom Standard weg auf einen
    RAND-Wert des Suchraums gezogen hat -- dann deutet der Trend an, dass es
    in dieser Richtung evtl. noch weiter besser wird. Kategorische Parameter
    (Wahrnehmung, Aktivierung) und Parameter, die beim Standard geblieben
    sind, werden bewusst NICHT gemeldet (ergaebe nur Rauschen: "Wahrnehmung
    noch niedriger" ist sinnlos, ein unveraenderter Wert ist gerade KEIN
    Randwert-Signal)."""
    tipps = []
    best = zustand["best_overrides"]
    default = DQNConfig()
    for param, werte in SUCHRAUM.items():
        aktuell = best.get(param, getattr(default, param))
        if isinstance(aktuell, (bool, str, tuple, list)):
            continue                       # nur echte Zahlen-Parameter
        if aktuell == getattr(default, param):
            continue                       # nicht aktiv weggetunt -> kein Signal
        if len(werte) < 2 or aktuell not in werte:
            continue
        idx = werte.index(aktuell)
        if idx == len(werte) - 1:
            tipps.append(
                f"`{param}` = {aktuell!r} ist der HOECHSTE getestete Wert und "
                "wurde aktiv dorthin gezogen -- evtl. im Suchraum noch hoeher testen.")
        elif idx == 0:
            tipps.append(
                f"`{param}` = {aktuell!r} ist der NIEDRIGSTE getestete Wert und "
                "wurde aktiv dorthin gezogen -- evtl. im Suchraum noch niedriger testen.")
    if not tipps:
        tipps.append("Keine Rand-Auffaelligkeiten -- der aktuelle Suchraum "
                     "scheint gut abgedeckt.")
    return tipps


def schreibe_abschlussbericht(ordner: str, zustand: dict, status: str) -> str:
    pfad = os.path.join(ordner, "abschlussbericht.md")

    # Ø Score je (Parameter, Wert) ueber ALLE Messungen -- das "Gelernte".
    pro_wert: dict[str, dict[str, list[float]]] = {}
    for k in zustand["historie"]:
        if k["mittel"] is None:
            continue
        pro_wert.setdefault(k["parameter"], {}).setdefault(
            str(k["neuer_wert"]), []).append(k["mittel"])

    # Fertiger, copy-paste-faehiger Startbefehl. Der Wert-Teil kommt als
    # repr (String -> mit Anfuehrungszeichen, Tupel -> "(256, 128)"), das
    # GANZE feld=wert wird in DOPPELTE Anfuehrungszeichen gesetzt -- so ist
    # es fuer alle Typen gueltige Shell (die einfachen Anfuehrungszeichen
    # eines String-Werts liegen dann sauber INNEN).
    best_cmd = "python train_dqn.py --headless 480"
    for feld, wert in zustand["best_overrides"].items():
        w = tuple(wert) if isinstance(wert, list) else wert
        best_cmd += f' \\\n    --override "{feld}={w!r}"'

    zeilen = [
        "# Auto-Tuner Abschlussbericht",
        f"Erzeugt: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Status: **{status}**",
        "",
        f"Durchlaeufe: {zustand['durchlauf']}  ·  "
        f"Kandidaten gemessen: {zustand['kandidaten_gesamt']}  ·  "
        f"angenommene Verbesserungen: {zustand['verbesserungen']}  ·  "
        f"Trainings-Laeufe insgesamt: {zustand['laeufe_gesamt']}",
        "",
        f"## Beste gefundene Konfiguration — Score {zustand['best_score']:.2f}",
        "```",
        best_cmd,
        "```",
        "",
        "## Verlauf der Bestmarke",
    ]
    for eintrag in zustand["verlauf"]:
        zeilen.append(f"- {eintrag}")

    zeilen += ["", "## Gelerntes je Parameter (Ø Score aller Messungen mit diesem Wert)",
               "", "Der mit ✓ markierte Wert steht in der besten Config."]
    best = zustand["best_overrides"]
    for param in sorted(pro_wert):
        zeilen.append(f"\n**{param}**")
        aktuell = best.get(param, getattr(DQNConfig(), param))
        werte = pro_wert[param]
        for wert, scores in sorted(werte.items(),
                                   key=lambda kv: -sum(kv[1]) / len(kv[1])):
            mittel = sum(scores) / len(scores)
            marke = " ✓" if str(
                tuple(aktuell) if isinstance(aktuell, (list, tuple)) else aktuell
            ) == wert else ""
            zeilen.append(f"- {wert}: Ø {mittel:.2f}  ({len(scores)} Messung(en)){marke}")

    # Top-Kandidaten insgesamt
    gemessen = [k for k in zustand["historie"] if k["mittel"] is not None]
    gemessen.sort(key=lambda k: -k["mittel"])
    zeilen += ["", "## Top 15 Einzel-Kandidaten (bester Ø zuerst)"]
    for k in gemessen[:15]:
        zeilen.append(
            f"- {k['mittel']:.2f} — {k['parameter']}={k['neuer_wert']!r} "
            f"(Durchlauf {k.get('durchlauf', '?')})")

    zeilen += ["", "## Empfehlungen fuer die naechste Runde"]
    for t in _empfehlungen(zustand):
        zeilen.append(f"- {t}")
    zeilen += [
        "",
        "Weitersuchen (nimmt den Zustand exakt hier wieder auf):",
        "```",
        f"python auto_tuner.py --fortsetzen {os.path.join(ordner, 'zustand.json')}",
        "```",
    ]

    with open(pfad, "w", encoding="utf-8") as fh:
        fh.write("\n".join(zeilen) + "\n")

    # Maschinenlesbare Kurzfassung fuer spaetere Auswertung/Weiterverarbeitung.
    with open(os.path.join(ordner, "auswertung.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "status": status,
            "best_overrides": zustand["best_overrides"],
            "best_score": zustand["best_score"],
            "durchlaeufe": zustand["durchlauf"],
            "laeufe_gesamt": zustand["laeufe_gesamt"],
            "gelerntes_je_wert": {
                p: {w: sum(s) / len(s) for w, s in werte.items()}
                for p, werte in pro_wert.items()
            },
        }, fh, indent=2, default=str, ensure_ascii=False)

    return pfad


# ------------------------------------------------------------------------- #
# Hauptschleife
# ------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomer DQN-Config-Tuner")
    parser.add_argument("--minuten", type=float, default=10.0,
                        help="Trainingszeit je Lauf (Standard 10 -- Plateau "
                             "wird meist in 5-10 Min erreicht)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds je Kandidat-Messung (Standard 1 -- die "
                             "Robustheit kommt ueber die Durchlaeufe, die je "
                             "einen ANDEREN Seed benutzen; 2+ = strenger, langsamer)")
    parser.add_argument("--max-stunden", type=float, default=48.0,
                        help="Sicherheits-Obergrenze; normal stoppt der Tuner "
                             "selbst bei Konvergenz")
    parser.add_argument("--marge", type=float, default=0.02,
                        help="Mindest-Verbesserung (relativ), um einen Wert zu uebernehmen")
    parser.add_argument("--max-kandidaten", type=int, default=0,
                        help="0 = unbegrenzt (nur fuer Tests)")
    parser.add_argument("--fortsetzen", metavar="ZUSTAND_JSON",
                        help="einen frueheren Tuner-Lauf fortsetzen")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    n_seeds = max(1, min(args.seeds, len(SEED_POOL)))

    if args.fortsetzen:
        with open(args.fortsetzen, encoding="utf-8") as fh:
            zustand = json.load(fh)
        if "hidden" in zustand["best_overrides"]:
            zustand["best_overrides"]["hidden"] = tuple(zustand["best_overrides"]["hidden"])
        ordner = os.path.dirname(os.path.abspath(args.fortsetzen))
        protokoll = Protokoll(ordner)
        protokoll.notiz(f"FORTSETZUNG: Bestmarke {zustand['best_score']:.2f} "
                        f"(Durchlauf {zustand['durchlauf']})")
    else:
        ordner = os.path.join(LOG_DIR, f"autotuner-{time.strftime('%Y%m%d-%H%M%S')}")
        protokoll = Protokoll(ordner)

        # Modell-Dateien sichern (der Tuner ueberschreibt runbest_*.pt laufend).
        sicherung = os.path.join(ordner, "modell-sicherung")
        os.makedirs(sicherung, exist_ok=True)
        if os.path.isdir(MODEL_DIR):
            for name in os.listdir(MODEL_DIR):
                quelle = os.path.join(MODEL_DIR, name)
                if os.path.isfile(quelle):
                    shutil.copy2(quelle, os.path.join(sicherung, name))
        protokoll.notiz(f"Modell-Sicherung nach {sicherung}")

        zustand = {
            "best_overrides": dict(BASIS_ABWEICHUNGEN),
            "best_score": 0.0,
            "historie": [],
            "verlauf": [],
            "durchlauf": 0,
            "pass_queue": [],          # noch offene Parameter im aktuellen Durchlauf
            "pass_verbessert": False,  # gab es in diesem Durchlauf eine Verbesserung?
            "kandidaten_gesamt": 0,
            "verbesserungen": 0,
            "laeufe_gesamt": 0,
            "seeds": n_seeds,
            "minuten": args.minuten,
        }
        protokoll.notiz(f"START: Basis {BASIS_ABWEICHUNGEN}, {args.minuten:g} Min/Lauf, "
                        f"{n_seeds} Seed(s)/Kandidat")

    start = time.time()
    status = "abgebrochen"

    def messe(overrides: dict, durchlauf: int, seeds: list[int]) -> float | None:
        """Misst eine Config ueber die gegebenen Seeds; Ø der Laufscores.
        None, wenn ein Lauf fehlschlaegt. Protokolliert jeden Lauf."""
        scores = []
        for seed in seeds:
            erg = run_training(overrides, seed, args.minuten)
            erg["seed"] = seed
            protokoll.lauf(durchlauf, overrides, erg)
            zustand["laeufe_gesamt"] += 1
            if not erg["ok"]:
                return None
            scores.append(erg["score"])
        return sum(scores) / len(scores)

    try:
        while True:
            if _STOP["flag"]:
                status = "per Strg+C beendet"
                break
            if (time.time() - start) / 3600 >= args.max_stunden:
                status = "Zeit-Obergrenze (--max-stunden) erreicht"
                protokoll.notiz(f"STOP: {status}.")
                break
            if args.max_kandidaten and zustand["kandidaten_gesamt"] >= args.max_kandidaten:
                status = "--max-kandidaten erreicht (Testmodus)"
                protokoll.notiz(f"STOP: {status}.")
                break

            # ---- Neuen Durchlauf beginnen, falls der alte leer ist -------- #
            if not zustand["pass_queue"]:
                if zustand["durchlauf"] > 0 and not zustand["pass_verbessert"]:
                    status = "KONVERGIERT (kompletter Durchlauf ohne Verbesserung)"
                    protokoll.notiz(
                        f"KONVERGENZ bestaetigt: Durchlauf {zustand['durchlauf']} "
                        "hat KEINEN Parameter mehr verbessert -- Optimum "
                        f"gefunden (Score {zustand['best_score']:.2f}). Tuner "
                        "schaltet sich ab.")
                    break
                zustand["durchlauf"] += 1
                params = list(SUCHRAUM.keys())
                random.Random(zustand["durchlauf"]).shuffle(params)
                zustand["pass_queue"] = params
                zustand["pass_verbessert"] = False

                # Seed(-block) fuer DIESEN Durchlauf -- rotiert, damit jeder
                # Wert ueber die Durchlaeufe verschiedene Seeds sieht.
                p = zustand["durchlauf"] - 1
                seeds = [SEED_POOL[(p * n_seeds + i) % len(SEED_POOL)]
                         for i in range(n_seeds)]
                zustand["_pass_seeds"] = seeds

                # Bestmarke frisch nachmessen (Waerme-Drift-Anker fuer diesen
                # Durchlauf), mit denselben Seeds wie die Kandidaten.
                neu = messe(zustand["best_overrides"], zustand["durchlauf"], seeds)
                if neu is not None:
                    if zustand["durchlauf"] > 1:
                        protokoll.notiz(
                            f"Durchlauf {zustand['durchlauf']} startet — Bestmarke "
                            f"frisch nachgemessen: {zustand['best_score']:.2f} -> {neu:.2f} "
                            f"(Seeds {seeds})")
                    zustand["best_score"] = neu
                    if not zustand["verlauf"]:
                        zustand["verlauf"].append(
                            f"Basis (Durchlauf 1, Seed {seeds}): {neu:.2f}")
                protokoll.zustand(zustand)
                if _STOP["flag"]:
                    status = "per Strg+C beendet"
                    break

            # ---- Naechsten Parameter des Durchlaufs abarbeiten ------------ #
            param = zustand["pass_queue"][0]
            seeds = zustand["_pass_seeds"]
            best_wert = None
            best_wert_score = zustand["best_score"]

            for wert in _alternativen(param, zustand["best_overrides"]):
                if _STOP["flag"] or (time.time() - start) / 3600 >= args.max_stunden:
                    break
                kandidat = dict(zustand["best_overrides"])
                kandidat[param] = wert
                zustand["kandidaten_gesamt"] += 1
                nr = zustand["kandidaten_gesamt"]
                protokoll.notiz(
                    f"[D{zustand['durchlauf']}] Kandidat #{nr}: {param} -> {wert!r} "
                    f"(Bestmarke {zustand['best_score']:.2f})")
                mittel = messe(kandidat, zustand["durchlauf"], seeds)
                zustand["historie"].append({
                    "nr": nr, "durchlauf": zustand["durchlauf"],
                    "parameter": param, "neuer_wert": wert, "mittel": mittel,
                })
                anzeige = "FEHLGESCHLAGEN" if mittel is None else f"{mittel:.2f}"
                protokoll.notiz(f"    Ergebnis: {anzeige}")
                if mittel is not None and mittel > best_wert_score:
                    best_wert, best_wert_score = wert, mittel
                protokoll.zustand(zustand)

            # ---- Besten Wert dieses Parameters ggf. uebernehmen ---------- #
            if (best_wert is not None
                    and best_wert_score > zustand["best_score"] * (1 + args.marge)):
                alt = zustand["best_score"]
                zustand["best_overrides"][param] = best_wert
                zustand["best_score"] = best_wert_score
                zustand["verbesserungen"] += 1
                zustand["pass_verbessert"] = True
                zustand["verlauf"].append(
                    f"D{zustand['durchlauf']}: {param}={best_wert!r} ANGENOMMEN "
                    f"({alt:.2f} -> {best_wert_score:.2f})")
                protokoll.notiz(
                    f"  => {param} = {best_wert!r} ANGENOMMEN "
                    f"({alt:.2f} -> {best_wert_score:.2f})")

            zustand["pass_queue"].pop(0)
            protokoll.zustand(zustand)

    except KeyboardInterrupt:
        status = "per Strg+C beendet"
        protokoll.notiz("STOP: per Strg+C beendet.")

    protokoll.notiz(f"FERTIG ({status}). Beste Config Score {zustand['best_score']:.2f}.")
    protokoll.zustand(zustand)
    pfad = schreibe_abschlussbericht(ordner, zustand, status)
    print(f"\nAbschlussbericht: {pfad}", flush=True)
    print(f"Beste Config (Score {zustand['best_score']:.2f}): "
          f"{json.dumps(zustand['best_overrides'], default=str, ensure_ascii=False)}",
          flush=True)


if __name__ == "__main__":
    main()
