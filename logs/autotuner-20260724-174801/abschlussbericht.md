# Auto-Tuner Abschlussbericht
Erzeugt: 2026-07-25 20:40:44
Status: **per Strg+C beendet**

Durchlaeufe: 4  ·  Kandidaten gemessen: 157  ·  angenommene Verbesserungen: 10  ·  Trainings-Laeufe insgesamt: 161

## Beste gefundene Konfiguration — Score 144.70
```
python train_dqn.py --headless 480 \
    --override "perception='rich_grid9'" \
    --override "spiegel_lernen=False" \
    --override "fruit_count=10" \
    --override "target_update=500" \
    --override "curriculum_anteil=1.0" \
    --override "n_step=5" \
    --override "pfad_fokus=0.7" \
    --override "gamma=0.99" \
    --override "batch_size=512"
```

## Verlauf der Bestmarke
- Basis (Durchlauf 1, Seed [11]): 65.65
- D1: perception='rich_grid9' ANGENOMMEN (65.65 -> 67.75)
- D1: spiegel_lernen=False ANGENOMMEN (67.75 -> 70.15)
- D1: fruit_count=10 ANGENOMMEN (70.15 -> 106.90)
- D1: target_update=2000 ANGENOMMEN (106.90 -> 113.15)
- D1: curriculum_anteil=1.0 ANGENOMMEN (113.15 -> 120.65)
- D1: n_step=5 ANGENOMMEN (120.65 -> 123.10)
- D2: target_update=500 ANGENOMMEN (120.40 -> 122.85)
- D2: pfad_fokus=0.7 ANGENOMMEN (122.85 -> 132.35)
- D2: gamma=0.99 ANGENOMMEN (132.35 -> 136.65)
- D3: batch_size=512 ANGENOMMEN (146.20 -> 152.45)

## Gelerntes je Parameter (Ø Score aller Messungen mit diesem Wert)

Der mit ✓ markierte Wert steht in der besten Config.

**activation**
- tanh: Ø 111.28  (4 Messung(en))

**balance_anteil**
- 0.5: Ø 132.26  (4 Messung(en))
- 0.0: Ø 127.21  (4 Messung(en))

**batch_size**
- 512: Ø 132.45  (3 Messung(en)) ✓
- 128: Ø 125.62  (3 Messung(en))

**curriculum_anteil**
- 0.25: Ø 124.20  (2 Messung(en))
- 1.0: Ø 120.65  (1 Messung(en)) ✓
- 0.5: Ø 117.32  (3 Messung(en))
- 0.0: Ø 116.56  (4 Messung(en))

**eps_decay_steps**
- 40000: Ø 124.37  (3 Messung(en))
- 150000: Ø 111.85  (3 Messung(en))

**eps_end**
- 0.05: Ø 126.94  (4 Messung(en))

**fruit_count**
- 10: Ø 106.90  (1 Messung(en)) ✓
- 5: Ø 103.44  (4 Messung(en))
- 3: Ø 84.62  (3 Messung(en))
- 1: Ø 63.59  (4 Messung(en))

**gamma**
- 0.97: Ø 124.70  (2 Messung(en))
- 0.99: Ø 102.40  (2 Messung(en)) ✓
- 0.95: Ø 101.44  (4 Messung(en))

**hidden**
- (512, 256): Ø 137.93  (4 Messung(en))
- (256, 256): Ø 133.80  (4 Messung(en))
- (128, 128): Ø 133.16  (4 Messung(en))

**learning_rate**
- 0.002: Ø 113.40  (4 Messung(en))
- 0.0005: Ø 112.66  (4 Messung(en))

**n_step**
- 3: Ø 131.98  (3 Messung(en))
- 5: Ø 123.10  (1 Messung(en)) ✓
- 1: Ø 116.53  (4 Messung(en))

**num_games**
- 8: Ø 115.69  (4 Messung(en))
- 24: Ø 114.98  (4 Messung(en))

**perception**
- rich_grid7: Ø 126.43  (3 Messung(en))
- rich_grid5: Ø 109.92  (4 Messung(en))
- rich: Ø 106.55  (4 Messung(en))
- rich_grid9: Ø 67.75  (1 Messung(en)) ✓

**pfad_fokus**
- 0.0: Ø 143.75  (2 Messung(en))
- 0.25: Ø 127.35  (4 Messung(en))
- 0.5: Ø 124.61  (4 Messung(en))
- 0.7: Ø 118.85  (2 Messung(en)) ✓
- 1.0: Ø 92.20  (4 Messung(en))

**pfad_fokus_bonus**
- 0.02: Ø 118.26  (4 Messung(en))
- 0.1: Ø 116.99  (4 Messung(en))

**prioritized**
- True: Ø 113.09  (4 Messung(en))

**reward_death**
- -20.0: Ø 104.90  (4 Messung(en))

**reward_step**
- 0.0: Ø 112.36  (4 Messung(en))

**spiegel_lernen**
- True: Ø 105.43  (3 Messung(en))
- False: Ø 70.15  (1 Messung(en)) ✓

**target_update**
- 1000: Ø 133.28  (3 Messung(en))
- 2000: Ø 129.58  (3 Messung(en))
- 500: Ø 117.12  (2 Messung(en)) ✓

**train_every**
- 2: Ø 108.03  (3 Messung(en))

## Top 15 Einzel-Kandidaten (bester Ø zuerst)
- 152.45 — batch_size=512 (Durchlauf 3)
- 147.65 — hidden=(512, 256) (Durchlauf 3)
- 147.35 — hidden=(128, 128) (Durchlauf 4)
- 146.80 — hidden=(512, 256) (Durchlauf 4)
- 146.65 — pfad_fokus=0.0 (Durchlauf 4)
- 146.20 — pfad_fokus_bonus=0.02 (Durchlauf 4)
- 145.50 — pfad_fokus_bonus=0.02 (Durchlauf 3)
- 144.30 — hidden=(256, 256) (Durchlauf 3)
- 143.50 — pfad_fokus_bonus=0.1 (Durchlauf 3)
- 143.40 — balance_anteil=0.5 (Durchlauf 3)
- 142.00 — reward_step=0.0 (Durchlauf 4)
- 141.90 — learning_rate=0.002 (Durchlauf 4)
- 141.10 — eps_end=0.05 (Durchlauf 3)
- 140.95 — prioritized=True (Durchlauf 3)
- 140.85 — pfad_fokus=0.0 (Durchlauf 3)

## Empfehlungen fuer die naechste Runde
- `gamma` = 0.99 ist der HOECHSTE getestete Wert und wurde aktiv dorthin gezogen -- evtl. im Suchraum noch hoeher testen.
- `n_step` = 5 ist der HOECHSTE getestete Wert und wurde aktiv dorthin gezogen -- evtl. im Suchraum noch hoeher testen.
- `batch_size` = 512 ist der HOECHSTE getestete Wert und wurde aktiv dorthin gezogen -- evtl. im Suchraum noch hoeher testen.
- `fruit_count` = 10 ist der HOECHSTE getestete Wert und wurde aktiv dorthin gezogen -- evtl. im Suchraum noch hoeher testen.
- `target_update` = 500 ist der NIEDRIGSTE getestete Wert und wurde aktiv dorthin gezogen -- evtl. im Suchraum noch niedriger testen.
- `curriculum_anteil` = 1.0 ist der HOECHSTE getestete Wert und wurde aktiv dorthin gezogen -- evtl. im Suchraum noch hoeher testen.

Weitersuchen (nimmt den Zustand exakt hier wieder auf):
```
python auto_tuner.py --fortsetzen /Users/lucamuller/Documents/snake-ai/logs/autotuner-20260724-174801/zustand.json
```
