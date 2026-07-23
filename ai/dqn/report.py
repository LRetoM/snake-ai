"""Post-Run-Report (TRAININGSPLAN.md Phase 0.1).

Nach JEDEM Trainingslauf (Esc im Fenster, Fenster schliessen, Headless
fertig, auch Strg+C) automatisch eine JSON- + Markdown-Auswertung schreiben.
Ohne das muesste man nach jedem Lauf raten, was passiert ist -- mit dem
Report SIEHT man es: an welcher Schlangenlaenge stirbt der Bot, woran, ist
das ein Plateau, hat sich das Netz selbst ueberschaetzt.

Die wichtigste Auswertung ist "Todesursachen nach Schlangenlaenge" -- sie
zeigt exakt, ab welcher Laenge Selbstkollisionen dominieren und ob eine
Aenderung (Wahrnehmung, Laengen-Balance, ...) genau dort etwas bewirkt.
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


def _death_length_buckets(episode_log: list[tuple], grid_cols: int, grid_rows: int):
    """Baut die Todesursachen-/Effizienz-Tabelle nach Laengen-Eimer plus das
    Todes-Positionsraster fuer Selbstkollisionen."""
    buckets: dict[str, dict] = {}
    death_grid = np.zeros((grid_rows, grid_cols), dtype=np.int64)
    score_hist: dict[str, int] = {}

    for length, cause, head_x, head_y, score, steps in episode_log:
        b = _length_bucket(length)
        entry = buckets.setdefault(b, {
            "gesamt": 0, "wall": 0, "self": 0, "starvation": 0, "won": 0,
            "zuege_summe": 0, "score_summe": 0,
        })
        entry["gesamt"] += 1
        entry[cause] = entry.get(cause, 0) + 1
        entry["zuege_summe"] += steps
        entry["score_summe"] += score

        if cause == "self" and 0 <= head_y < grid_rows and 0 <= head_x < grid_cols:
            death_grid[head_y, head_x] += 1

        s_bucket = (max(0, score) // 10) * 10
        key = f"{s_bucket}-{s_bucket + 9}"
        score_hist[key] = score_hist.get(key, 0) + 1

    todeslaengen = {}
    for b, e in sorted(buckets.items(), key=lambda kv: _bucket_sort_key(kv[0])):
        gesamt = e["gesamt"]
        todeslaengen[b] = {
            "episoden": gesamt,
            "wand_pct": round(100 * e.get("wall", 0) / gesamt, 1),
            "selbst_pct": round(100 * e.get("self", 0) / gesamt, 1),
            "verhungert_pct": round(100 * e.get("starvation", 0) / gesamt, 1),
            "sieg_pct": round(100 * e.get("won", 0) / gesamt, 1),
            "zuege_pro_frucht": (round(e["zuege_summe"] / e["score_summe"], 2)
                                 if e["score_summe"] else None),
        }
    score_hist = dict(sorted(score_hist.items(), key=lambda kv: int(kv[0].split("-")[0])))
    return todeslaengen, score_hist, death_grid


def _diagnosen(todeslaengen: dict, eval_curve: list[float], eval_best: float,
                loss_history: list[float]) -> list[str]:
    """Automatische Textbefunde -- reine DIAGNOSE, kein Eingriff. Genau der
    Mittelweg, den Luca wollte: erkennen + melden, nicht selbst anpassen."""
    diagnosen = []

    for b, d in todeslaengen.items():
        lo = _bucket_sort_key(b)
        if lo >= 30 and d["episoden"] >= 10 and d["selbst_pct"] > 60:
            diagnosen.append(
                f"Selbstkollision {d['selbst_pct']}% im Laengen-Eimer {b} "
                f"({d['episoden']} Episoden) -- Endspiel-Einsperren dominiert dort."
            )

    if len(eval_curve) >= 5 and eval_best > 0:
        recent = eval_curve[-5:]
        if all(v < eval_best * 0.95 for v in recent):
            diagnosen.append(
                f"Pruefung stagniert: die letzten 5 Pruefungen "
                f"({[round(v, 1) for v in recent]}) liegen alle unter 95% "
                f"des Bestwerts ({eval_best:.1f}) -- moeglicherweise ein Plateau."
            )

    if len(loss_history) >= 6:
        third = max(1, len(loss_history) // 3)
        early = float(np.mean(loss_history[:third]))
        late = float(np.mean(loss_history[-third:]))
        if early > 0 and late > early * 1.1:
            diagnosen.append(
                f"Loss steigt ueber das letzte Drittel des Laufs "
                f"({early:.3f} -> {late:.3f}) -- Lernrate/Stabilitaet pruefen."
            )

    if not diagnosen:
        diagnosen.append("Keine auffaelligen Befunde.")
    return diagnosen


def build_report(trainer) -> dict:
    """Baut das Report-Dict aus einem MultiGameTrainer. Reine Auswertung
    bereits gesammelter Daten -- veraendert am Trainer nichts."""
    cfg = trainer.cfg
    elapsed = time.time() - trainer.started_at

    todeslaengen, score_hist, death_grid = _death_length_buckets(
        trainer.episode_log, cfg.grid_cols, cfg.grid_rows)

    q_kalibrierung = None
    if trainer.q_calibration_log:
        qs = np.array([q for q, _ in trainer.q_calibration_log])
        actual = np.array([s for _, s in trainer.q_calibration_log])
        q_kalibrierung = {
            "mittlerer_start_q": round(float(qs.mean()), 2),
            "mittlerer_erzielter_score": round(float(actual.mean()), 2),
            "differenz_q_minus_score": round(float(qs.mean() - actual.mean()), 2),
            "anzahl_partien": int(len(qs)),
        }

    diagnosen = _diagnosen(todeslaengen, trainer.eval_curve, trainer.eval_best,
                           trainer.loss_history)

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
            "bestwert": round(trainer.eval_best, 2),
            "champion_datei": trainer.champion_path,
            "champion_neu_in_diesem_lauf": trainer.champion_path is not None,
            "brett_transfer_von": trainer.board_transfer_from,
        },
        "todesursachen_nach_laenge": todeslaengen,
        "score_histogramm": score_hist,
        "todes_positionen_selbst": death_grid.tolist(),
        "q_kalibrierung": q_kalibrierung,
        "meilenstein_protokoll": trainer.milestone_log,
        "diagnosen": diagnosen,
    }


def _render_markdown(r: dict) -> str:
    lines = [
        f"# Trainings-Report -- Brett {r['brett']} ({r['wahrnehmung']})",
        f"Erzeugt: {r['erzeugt_am']}",
        "",
        f"Laufzeit: {r['laufzeit_sekunden'] / 60:.1f} Min · "
        f"Ticks: {r['ticks']:,} · Episoden: {r['episoden']:,} · "
        f"Zuege/s: {r['zuege_pro_sekunde']}".replace(",", "."),
        "",
        "## Pruefung",
        f"- Bestwert: **{r['pruefung']['bestwert']}**",
        f"- Champion-Datei: {r['pruefung']['champion_datei'] or '(keine gespeichert)'}",
        f"- Neuer Champion in diesem Lauf: "
        f"{'ja' if r['pruefung']['champion_neu_in_diesem_lauf'] else 'nein'}",
    ]
    if r["pruefung"]["brett_transfer_von"]:
        fc, fr = r["pruefung"]["brett_transfer_von"]
        lines.append(f"- Brett-Transfer von: {fc}x{fr}")

    lines += [
        "", "## Todesursachen nach Schlangenlaenge (die Kernauswertung)", "",
        "| Laenge | Episoden | Wand% | Selbst% | Verhungert% | Sieg% | Zuege/Frucht |",
        "|---|---|---|---|---|---|---|",
    ]
    for b, d in r["todesursachen_nach_laenge"].items():
        lines.append(
            f"| {b} | {d['episoden']} | {d['wand_pct']} | {d['selbst_pct']} | "
            f"{d['verhungert_pct']} | {d['sieg_pct']} | {d['zuege_pro_frucht']} |"
        )

    if r["score_histogramm"]:
        lines += ["", "## Score-Verteilung", "", "| Score | Episoden |", "|---|---|"]
        for b, n in r["score_histogramm"].items():
            lines.append(f"| {b} | {n} |")

    if r["q_kalibrierung"]:
        q = r["q_kalibrierung"]
        lines += [
            "", "## Q-Kalibrierung (Selbsteinschaetzung vs. Realitaet)",
            f"- Mittlerer vorhergesagter Start-Q: {q['mittlerer_start_q']}",
            f"- Mittlerer tatsaechlicher Score: {q['mittlerer_erzielter_score']}",
            f"- Differenz (Q - Score): {q['differenz_q_minus_score']} "
            "(deutlich positiv = Netz ueberschaetzt sich selbst)",
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
