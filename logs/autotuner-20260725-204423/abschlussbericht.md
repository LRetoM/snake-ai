# Auto-Tuner Abschlussbericht
Erzeugt: 2026-07-25 20:44:31
Status: **per Strg+C beendet**

Durchlaeufe: 1  ·  Kandidaten gemessen: 0  ·  angenommene Verbesserungen: 0  ·  Trainings-Laeufe insgesamt: 1

## Beste gefundene Konfiguration — Score 5.60
```
python train_dqn.py --headless 480 \
    --override "perception='rich_grid7'"
```

Trainierte Gewichte dieser genauen Konfiguration: `/Users/lucamuller/Documents/snake-ai/logs/autotuner-20260725-204423/checkpoints/D1_basis.pt` -- direkt ladbar, kein erneutes Training noetig, um sie anzuschauen (z.B. mit watch_ai.py als Vorlage).

## Verlauf der Bestmarke
- Basis (Durchlauf 1, Seed [11]): 5.60 [Gewichte: /Users/lucamuller/Documents/snake-ai/logs/autotuner-20260725-204423/checkpoints/D1_basis.pt]

## Gelerntes je Parameter (Ø Score aller Messungen mit diesem Wert)

Der mit ✓ markierte Wert steht in der besten Config.

## Top 15 Einzel-Kandidaten (bester Ø zuerst)

## Empfehlungen fuer die naechste Runde
- Keine Rand-Auffaelligkeiten -- der aktuelle Suchraum scheint gut abgedeckt.

Weitersuchen (nimmt den Zustand exakt hier wieder auf):
```
python auto_tuner.py --fortsetzen /Users/lucamuller/Documents/snake-ai/logs/autotuner-20260725-204423/zustand.json
```
