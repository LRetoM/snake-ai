"""Post-Run-Report (TRAININGSPLAN.md Phase 0.1).

Nach JEDEM Trainingslauf (Esc im Fenster, Fenster schliessen, Headless
fertig, auch Strg+C) automatisch eine JSON- + Markdown-Auswertung schreiben.
Ohne das muesste man nach jedem Lauf raten, was passiert ist -- mit dem
Report SIEHT man es: an welcher Schlangenlaenge stirbt der Bot, woran, ist
das ein Plateau, wie ehrlich schaetzt sich das Netz selbst ein.

Die wichtigste Auswertung ist "Todesursachen nach Schlangenlaenge" -- sie
zeigt exakt, ab welcher Laenge Selbstkollisionen dominieren und ob eine
Aenderung (Wahrnehmung, Laengen-Balance, Curriculum, ...) genau dort etwas
bewirkt.

Grundsaetze dieses Reports (aus den Fehlern der ersten Fassung gelernt):
- Alle Diagnosen messen am Niveau DIESES Laufs (eval_best_run), nie an der
  Champion-Schutzschwelle der Datei -- die kann von einem frueheren Lauf mit
  voellig anderer Wahrnehmung stammen und hat mit diesem Lauf nichts zu tun.
- Curriculum-Starts (Partien, die absichtlich lang beginnen) werden von
  natuerlich gestarteten Partien GETRENNT ausgewertet -- sonst verzerren sie
  jede Laengen-Statistik unbemerkt.
- Der Loss wird nicht absolut bewertet, sondern relativ zur Q-Skala: wenn
  die Q-Werte wachsen (weil der Bot laenger lebt und mehr einsammelt),
  waechst der absolute Loss zwangslaeufig mit -- das ist kein Problem,
  sondern Physik des Lernziels.
- Die Q-Kalibrierung sagt ehrlich, in WELCHE Richtung das Netz danebenliegt,
  und erklaert den systematischen Anteil (Q ist mit gamma abgezinst, der
  Score nicht -- bei langen Partien klafft das zwangslaeufig auseinander).
"""

from __future__ import annotations

import json
import time
from dataclasses import fields

import numpy as np

from ai.dqn.config import DQNConfig

LENGTH_BUCKET_SIZE = 10


def _length_bucket(length: int) -> str:
    """Rundet eine Schlangenlaenge auf einen 10er-Eimer ab ("0-9", "30-39",
    "90+" fuer alles ab 90)."""
    if length >= 90:
        return "90+"
    lo = (length // LENGTH_BUCKET_SIZE) * LENGTH_BUCKET_SIZE
    return f"{lo}-{lo + LENGTH_BUCKET_SIZE - 1}"


def _bucket_sort_key(bucket: str) -> int:
    return int(bucket.split("-")[0].replace("+", ""))


def _config_diff(cfg: DQNConfig) -> dict:
    """Nur die Felder, die vom Code-Standard abweichen -- macht auf einen
    Blick klar, was an DIESEM Lauf besonders eingestellt war."""
    default = DQNConfig()
    diff = {}
    for f in fields(DQNConfig):
        cur = getattr(cfg, f.name)
        base = getattr(default, f.name)
        if cur != base:
            diff[f.name] = {"lauf": cur, "standard": base}
    return diff


def _empty_bucket() -> dict:
    return {
        "gesamt": 0, "wall": 0, "self": 0, "starvation": 0, "won": 0,
        "zuege_summe": 0, "score_summe": 0,
    }


def _death_length_buckets(episode_log: list[tuple], grid_cols: int, grid_rows: int):
    """Baut die Todesursachen-/Effizienz-Tabellen nach Laengen-Eimer plus das
    Todes-Positionsraster fuer Selbstkollisionen.

    Curriculum-Starts werden GETRENNT gezaehlt: eine Partie, die absichtlich
    bei Laenge 40 beginnt, landet zwangslaeufig in hohen Laengen-Eimern --
    wuerde man sie mit natuerlich gestarteten mischen, saehe die Verteilung
    voellig anders aus, ohne dass der Bot anders spielt. Aeltere episode_log-
    Eintraege (6 Felder, ohne Curriculum-Flag) zaehlen als natuerlich.
    """
    natural: dict[str, dict] = {}
    curriculum: dict[str, dict] = {}
    death_grid = np.zeros((grid_rows, grid_cols), dtype=np.int64)
    score_hist: dict[str, int] = {}

    for entry in episode_log:
        length, cause, head_x, head_y, score, steps = entry[:6]
        von_curriculum = bool(entry[6]) if len(entry) > 6 else False

        b = _length_bucket(length)
        buckets = curriculum if von_curriculum else natural
        e = buckets.setdefault(b, _empty_bucket())
        e["gesamt"] += 1
        e[cause] = e.get(cause, 0) + 1
        e["zuege_summe"] += steps
        e["score_summe"] += score

        if cause == "self" and 0 <= head_y < grid_rows and 0 <= head_x < grid_cols:
            death_grid[head_y, head_x] += 1

        if not von_curriculum:
            s_bucket = (max(0, score) // 10) * 10
            key = f"{s_bucket}-{s_bucket + 9}"
            score_hist[key] = score_hist.get(key, 0) + 1

    def to_table(buckets: dict) -> dict:
        table = {}
        for b, e in sorted(buckets.items(), key=lambda kv: _bucket_sort_key(kv[0])):
            gesamt = e["gesamt"]
            table[b] = {
                "episoden": gesamt,
                "wand_pct": round(100 * e.get("wall", 0) / gesamt, 1),
                "selbst_pct": round(100 * e.get("self", 0) / gesamt, 1),
                "verhungert_pct": round(100 * e.get("starvation", 0) / gesamt, 1),
                "sieg_pct": round(100 * e.get("won", 0) / gesamt, 1),
                "zuege_pro_frucht": (round(e["zuege_summe"] / e["score_summe"], 2)
                                     if e["score_summe"] else None),
            }
        return table

    score_hist = dict(sorted(score_hist.items(), key=lambda kv: int(kv[0].split("-")[0])))
    return to_table(natural), to_table(curriculum), score_hist, death_grid


def _q_kalibrierung_bewertung(diff: float) -> str:
    """Richtungsbewusster Befund zur Q-Kalibrierung.

    Die fruehere Fassung schrieb pauschal "deutlich positiv = Netz
    ueberschaetzt sich" hinter JEDEN Wert -- auch hinter negative. Dazu
    kommt ein systematischer Anteil, den man kennen muss, bevor man den
    Wert interpretiert: der Start-Q ist eine mit gamma abgezinste Summe
    (ein Punkt in 50 Zuegen zaehlt nur ~0.97^50 = 22%), der Score dagegen
    zaehlt jede Frucht voll. Je laenger die Partien, desto weiter liegt Q
    deshalb GANZ OHNE Fehlkalibrierung unter dem Score.
    """
    if diff > 5:
        return ("Netz ueberschaetzt sich (sagt mehr voraus, als es erreicht) -- "
                "klassisches DQN-Warnsignal, double_dqn/Lernrate pruefen.")
    if diff < -5:
        return ("Q liegt unter dem Score -- zum Teil SYSTEMATISCH korrekt "
                "(Q ist mit gamma abgezinst, der Score nicht; bei langen "
                "Partien klafft das zwangslaeufig auseinander), also kein "
                "Alarmzeichen.")
    return "Gut kalibriert (Abweichung im Rahmen der Streuung)."


def _diagnosen(todeslaengen_natuerlich: dict, eval_curve: list[float],
                eval_best_run: float, eval_history: list[dict],
                champion_floor_info: dict | None, perception: str) -> list[str]:
    """Automatische Textbefunde -- reine DIAGNOSE, kein Eingriff. Genau der
    Mittelweg, den Luca wollte: erkennen + melden, nicht selbst anpassen.

    Alle Vergleiche laufen gegen das Niveau DIESES Laufs (eval_best_run) --
    die Champion-Schutzschwelle der Datei ist dafuer ungeeignet (kann von
    einem Lauf mit anderer Wahrnehmung stammen; genau dieser Fehler hat am
    2026-07-24 zweimal ein falsches "Plateau" gemeldet).
    """
    diagnosen = []

    for b, d in todeslaengen_natuerlich.items():
        lo = _bucket_sort_key(b)
        if lo >= 30 and d["episoden"] >= 10 and d["selbst_pct"] > 60:
            diagnosen.append(
                f"Selbstkollision {d['selbst_pct']}% im Laengen-Eimer {b} "
                f"({d['episoden']} natuerlich gestartete Episoden) -- "
                "Endspiel-Einsperren dominiert dort."
            )

    if len(eval_curve) >= 5 and eval_best_run > 0:
        recent = eval_curve[-5:]
        if all(v < eval_best_run * 0.95 for v in recent):
            diagnosen.append(
                f"Pruefung stagniert: die letzten 5 Pruefungen "
                f"({[round(v, 1) for v in recent]}) liegen alle unter 95% "
                f"des Lauf-Bestwerts ({eval_best_run:.1f}) -- moeglicherweise ein Plateau."
            )

    # Loss relativ zur Q-Skala bewerten, nicht absolut: wachsende Q-Werte
    # (laengere Partien, mehr Belohnung im Umlauf) ziehen den absoluten Loss
    # zwangslaeufig mit hoch -- erst ein steigendes VERHAELTNIS Loss/Q ist
    # ein echtes Stabilitaets-Signal.
    with_loss = [h for h in eval_history
                 if h.get("loss") is not None and h.get("start_q")]
    if len(with_loss) >= 6:
        third = max(1, len(with_loss) // 3)
        def norm(h):
            return h["loss"] / max(1.0, abs(h["start_q"]))
        early = float(np.mean([norm(h) for h in with_loss[:third]]))
        late = float(np.mean([norm(h) for h in with_loss[-third:]]))
        if early > 0 and late > early * 1.3:
            diagnosen.append(
                f"Loss waechst schneller als die Q-Skala "
                f"(normiert {early:.4f} -> {late:.4f}) -- echtes "
                "Stabilitaets-Signal, Lernrate pruefen."
            )

    # Geschwindigkeit ueber den Lauf: haette den memory.py-Einbruch vom
    # 2026-07-24 (11k -> 3k Zuege/s) automatisch gemeldet.
    with_rate = [h for h in eval_history if h.get("zuege_pro_s")]
    if len(with_rate) >= 6:
        third = max(1, len(with_rate) // 3)
        early = float(np.mean([h["zuege_pro_s"] for h in with_rate[:third]]))
        late = float(np.mean([h["zuege_pro_s"] for h in with_rate[-third:]]))
        if early > 0 and late < early * 0.7:
            diagnosen.append(
                f"Geschwindigkeit faellt ueber den Lauf ({early:.0f} -> "
                f"{late:.0f} Zuege/s im Drittel-Vergleich) -- Performance-"
                "Problem im Trainingspfad, nicht im Bot."
            )

    if (champion_floor_info
            and champion_floor_info.get("wahrnehmung") not in (None, perception)):
        diagnosen.append(
            f"Champion-Schutzschwelle {champion_floor_info['wert']:.1f} stammt "
            f"von einem Lauf mit Wahrnehmung '{champion_floor_info['wahrnehmung']}' "
            f"(dieser Lauf: '{perception}') -- die Werte sind NICHT direkt "
            "vergleichbar; dieser Lauf kann die Champion-Datei erst ab dieser "
            "Schwelle ueberschreiben, sein Bestes liegt aber ohnehin in der "
            "Runbest-Datei."
        )

    if not diagnosen:
        diagnosen.append("Keine auffaelligen Befunde.")
    return diagnosen


def build_report(trainer) -> dict:
    """Baut das Report-Dict aus einem MultiGameTrainer. Reine Auswertung
    bereits gesammelter Daten -- veraendert am Trainer nichts."""
    cfg = trainer.cfg
    elapsed = time.time() - trainer.started_at

    (todeslaengen, todeslaengen_curriculum,
     score_hist, death_grid) = _death_length_buckets(
        trainer.episode_log, cfg.grid_cols, cfg.grid_rows)

    q_kalibrierung = None
    if trainer.q_calibration_log:
        qs = np.array([q for q, _ in trainer.q_calibration_log])
        actual = np.array([s for _, s in trainer.q_calibration_log])
        diff = float(qs.mean() - actual.mean())
        q_kalibrierung = {
            "mittlerer_start_q": round(float(qs.mean()), 2),
            "mittlerer_erzielter_score": round(float(actual.mean()), 2),
            "differenz_q_minus_score": round(diff, 2),
            "bewertung": _q_kalibrierung_bewertung(diff),
            "anzahl_partien": int(len(qs)),
        }

    eval_history = list(getattr(trainer, "eval_history", []))
    champion_floor_info = getattr(trainer, "champion_floor_info", None)
    eval_best_run = float(getattr(trainer, "eval_best_run", trainer.eval_best))

    diagnosen = _diagnosen(todeslaengen, trainer.eval_curve, eval_best_run,
                           eval_history, champion_floor_info, cfg.perception)

    curriculum_episoden = sum(d["episoden"] for d in todeslaengen_curriculum.values())
    natuerliche_episoden = sum(d["episoden"] for d in todeslaengen.values())

    return {
        "erzeugt_am": time.strftime("%Y-%m-%d %H:%M:%S"),
        "brett": f"{cfg.grid_cols}x{cfg.grid_rows}",
        "wahrnehmung": cfg.perception,
        "config_abweichungen_vom_standard": _config_diff(cfg),
        "laufzeit_sekunden": round(elapsed, 1),
        "ticks": trainer.total_steps,
        "episoden": trainer.total_episodes,
        "zuege_pro_sekunde": round(trainer.total_moves / elapsed, 1) if elapsed > 0 else None,
        "pruefung": {
            "verlauf": [round(v, 2) for v in trainer.eval_curve],
            "episoden_punkte": list(trainer.eval_points),
            "bestwert_dieses_laufs": round(eval_best_run, 2),
            "champion_schutzschwelle": round(trainer.eval_best, 2),
            "champion_schutzschwelle_info": champion_floor_info,
            "champion_datei": trainer.champion_path,
            "champion_neu_in_diesem_lauf": trainer.champion_path is not None,
            "runbest_datei": getattr(trainer, "runbest_saved", None),
            "brett_transfer_von": trainer.board_transfer_from,
            "historie": eval_history,
        },
        "curriculum": {
            "anteil_konfiguriert": getattr(cfg, "curriculum_anteil", 0.0),
            "gespeicherte_stellungen": len(getattr(trainer, "curriculum_snapshots", [])),
            "episoden_aus_curriculum": curriculum_episoden,
            "episoden_natuerlich": natuerliche_episoden,
            "traps_entfernt": getattr(trainer, "curriculum_traps_removed", 0),
        },
        "todesursachen_nach_laenge": todeslaengen,
        "todesursachen_nach_laenge_curriculum": todeslaengen_curriculum,
        "score_histogramm": score_hist,
        "todes_positionen_selbst": death_grid.tolist(),
        "q_kalibrierung": q_kalibrierung,
        "meilenstein_protokoll": trainer.milestone_log,
        "diagnosen": diagnosen,
    }


def _death_table_lines(table: dict, mit_effizienz: bool = True) -> list[str]:
    """Rendert eine Todesursachen-Tabelle. `mit_effizienz=False` fuer
    Curriculum-Starts: deren Score enthaelt die GEERBTE Startlaenge, die
    Zuege zaehlen aber erst ab dem Snapshot -- "Zuege/Frucht" waere dort
    eine sinnlose (absurd kleine) Zahl."""
    if mit_effizienz:
        lines = [
            "| Laenge | Episoden | Wand% | Selbst% | Verhungert% | Sieg% | Zuege/Frucht |",
            "|---|---|---|---|---|---|---|",
        ]
    else:
        lines = [
            "| Laenge | Episoden | Wand% | Selbst% | Verhungert% | Sieg% |",
            "|---|---|---|---|---|---|",
        ]
    for b, d in table.items():
        row = (f"| {b} | {d['episoden']} | {d['wand_pct']} | {d['selbst_pct']} | "
               f"{d['verhungert_pct']} | {d['sieg_pct']} |")
        if mit_effizienz:
            row += f" {d['zuege_pro_frucht']} |"
        lines.append(row)
    return lines


def _render_markdown(r: dict) -> str:
    p = r["pruefung"]
    lines = [
        f"# Trainings-Report -- Brett {r['brett']} ({r['wahrnehmung']})",
        f"Erzeugt: {r['erzeugt_am']}",
        "",
        f"Laufzeit: {r['laufzeit_sekunden'] / 60:.1f} Min · "
        f"Ticks: {r['ticks']:,} · Episoden: {r['episoden']:,} · "
        f"Zuege/s: {r['zuege_pro_sekunde']}".replace(",", "."),
        "",
        "## Pruefung",
        f"- Bestwert DIESES Laufs: **{p['bestwert_dieses_laufs']}**",
        f"- Champion-Schutzschwelle der Datei: {p['champion_schutzschwelle']}"
        + (f" (stammt von Wahrnehmung '{p['champion_schutzschwelle_info']['wahrnehmung']}')"
           if p.get("champion_schutzschwelle_info")
           and p["champion_schutzschwelle_info"].get("wahrnehmung") else ""),
        f"- Champion-Datei neu geschrieben: "
        f"{'ja -- ' + p['champion_datei'] if p['champion_neu_in_diesem_lauf'] else 'nein'}",
        f"- Bester Stand dieses Laufs gesichert: {p['runbest_datei'] or '(keine Pruefung gelaufen)'}",
    ]
    if p["brett_transfer_von"]:
        fc, fr = p["brett_transfer_von"]
        lines.append(f"- Brett-Transfer von: {fc}x{fr}")

    if p.get("historie"):
        lines += [
            "", "## Pruefungs-Historie (je Pruefung: Streuung, Tempo, Lernzustand)", "",
            "| Episode | Mittel | Min | Median | Max | Zuege/s | Loss | Start-Q |",
            "|---|---|---|---|---|---|---|---|",
        ]
        hist = p["historie"]
        # Bei sehr vielen Pruefungen nur Anfang + letzte 12 zeigen -- die
        # vollstaendige Liste steht immer im JSON.
        shown = hist if len(hist) <= 16 else hist[:3] + [None] + hist[-12:]
        for h in shown:
            if h is None:
                lines.append("| ... | | | | | | | |")
                continue
            lines.append(
                f"| {h['episode']} | {h['mittel']} | {h['min']} | {h['median']} | "
                f"{h['max']} | {h['zuege_pro_s']} | {h.get('loss', '')} | "
                f"{h.get('start_q', '')} |"
            )

    lines += ["", "## Todesursachen nach Schlangenlaenge (natuerlich gestartete Partien)", ""]
    lines += _death_table_lines(r["todesursachen_nach_laenge"])

    if r["todesursachen_nach_laenge_curriculum"]:
        c = r["curriculum"]
        lines += [
            "",
            f"## Curriculum-Starts ({c['episoden_aus_curriculum']} Partien, "
            f"getrennt ausgewertet -- sie beginnen absichtlich lang)", "",
        ]
        lines += _death_table_lines(r["todesursachen_nach_laenge_curriculum"],
                                    mit_effizienz=False)

    if r["curriculum"]["gespeicherte_stellungen"] or r["curriculum"]["traps_entfernt"]:
        c = r["curriculum"]
        lines += [
            "",
            f"Curriculum: Anteil {c['anteil_konfiguriert']}, "
            f"{c['gespeicherte_stellungen']} gespeicherte Stellungen, "
            f"{c['episoden_aus_curriculum']} Curriculum- vs. "
            f"{c['episoden_natuerlich']} natuerliche Episoden, "
            f"{c['traps_entfernt']} als aussichtslos entfernt "
            "(alle 3 ersten Zuege sterben fast immer schnell).",
        ]

    if r["score_histogramm"]:
        lines += ["", "## Score-Verteilung (nur natuerlich gestartete Partien)",
                  "", "| Score | Episoden |", "|---|---|"]
        for b, n in r["score_histogramm"].items():
            lines.append(f"| {b} | {n} |")

    if r["q_kalibrierung"]:
        q = r["q_kalibrierung"]
        lines += [
            "", "## Q-Kalibrierung (Selbsteinschaetzung vs. Realitaet)",
            f"- Mittlerer vorhergesagter Start-Q: {q['mittlerer_start_q']}",
            f"- Mittlerer tatsaechlicher Score: {q['mittlerer_erzielter_score']}",
            f"- Differenz (Q - Score): {q['differenz_q_minus_score']}",
            f"- Bewertung: {q['bewertung']}",
        ]

    if r["meilenstein_protokoll"]:
        lines += ["", "## Meilenstein-Aenderungen (deterministisch, siehe TRAININGSPLAN.md)"]
        for m in r["meilenstein_protokoll"]:
            lines.append(f"- Episode {m['episode']} (Pruefung {m['eval_best']:.1f}): {m['aenderung']}")

    if r["config_abweichungen_vom_standard"]:
        lines += ["", "## Config-Abweichungen vom Code-Standard"]
        for k, v in r["config_abweichungen_vom_standard"].items():
            lines.append(f"- `{k}`: {v['lauf']} (Standard: {v['standard']})")

    lines += ["", "## Diagnosen (automatisch, reine Meldung -- kein Eingriff)"]
    for d in r["diagnosen"]:
        lines.append(f"- {d}")

    return "\n".join(lines) + "\n"


def write_report(trainer, path_base: str) -> tuple[str, str]:
    """Schreibt `<path_base>-report.json` und `<path_base>-report.md`.

    Wird bei jedem Laufende aufgerufen (Esc im Fenster, Fenster schliessen,
    Headless fertig -- auch nach Strg+C, siehe try/finally in train_dqn.py).
    Ueberschreibt bestehende Dateien desselben Laufs anstandslos.
    """
    report = build_report(trainer)
    json_path = f"{path_base}-report.json"
    md_path = f"{path_base}-report.md"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(report))

    return json_path, md_path
