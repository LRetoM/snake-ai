# Trainingsplan & Umsetzungs-Spezifikation: Der Weg Richtung 100% Feldfüllung

Stand: 2026-07-23 (3. Fassung — Umsetzungs-Spez für Sonnet 5)
Basis-Messung: `rich_grid7`, Standard-Config, 20×20: Prüfung 73.9,
Champion 84.0, Feld 19.2%, Selbstkollision 52% aller Tode (66% der
letzten 100), Verhungern ~1%, Effizienz 9.8 Züge/Frucht, Kurve steigt noch.

Dieses Dokument hat drei Teile:
- **Teil A** — Getroffene Entscheidungen (gelten ab jetzt)
- **Teil B** — Bedienungsanleitung für Luca (was DU wann tust)
- **Teil C** — Umsetzungs-Spezifikation für Sonnet 5 (exakte Schritte,
  Dateien, Werte, Abnahmekriterien — in dieser Reihenfolge abarbeiten)

---

## UMSETZUNGSSTAND (2026-07-23, von Sonnet 5)

**Fertig, getestet, einsatzbereit:**
- **S0.1** Neue Defaults `grid_cols=17, grid_rows=15` in `ai/dqn/config.py`.
- **S0.2** `full_board` ist jetzt brettgrößen-dynamisch (`make_full_board_perception`,
  `get_perception(name, cols, rows)`). Kein Absturz mehr bei anderen Brettgrößen.
- **S0.3** Jede Brettgröße hat ihre eigene Champion-Datei
  (`champion_path()`/`resolve_champion_path()` in `ai/dqn/trainer.py`,
  Schema `dqn_champion_<cols>x<rows>.pt`). Alte `dqn_champion.pt` (20×20)
  wird als Legacy-Lesepfad weiter erkannt.
- **S0.4** Menü-Zeile "Brettgröße" (9×9 / 13×11 / 17×15 / 20×20) +
  Brett-Transfer beim Weitertrainieren (Gewichte + Ticks kommen mit,
  Rekorde starten neu) — siehe Bedienungsanleitung Teil B3.
- **S0.5** CLI `--brett SPALTENxZEILEN` in `train_dqn.py` (nur mit
  `--headless`; im Fenster die Menü-Zeile benutzen).
- **Phase 1.1** `activation`-Feld (relu/tanh) in Netz + Config + Checkpoint,
  DQN-Default jetzt `relu`, Neuroevolution bleibt `tanh`.
- **Phase 0.3 (Teil)**: `eval_episodes` 10→20 gesetzt. Dabei einen ECHTEN
  Bug fest gemacht und mitgefixt: `_eval_games` war auf `min(eval_episodes,
  16)` gedeckelt UND wurde bei einer nachträglichen Änderung von
  `cfg.eval_episodes` (z.B. die Abschluss-Prüfung mit 30 Partien in
  `train_dqn.py`) nie neu aufgebaut — beides lief bisher STILL SCHWEIGEND
  mit weniger Partien als eingestellt, ohne Fehlermeldung. Jetzt wächst
  `_eval_games` bei Bedarf nach.
- 2 vorbestehende Bugs aus einer früheren Session (Config-Reset bei
  `--weiter`, Champion-Überschreib-Schutz) sind weiterhin aktiv und wurden
  in diesem Umbau mitgezogen (jetzt pro Brett statt global).

**Verifiziert mit Smoke-Tests** (nicht Teil des Repos, im Chat ausgeführt):
Brett-Transfer 9×9→13×11 (Gewichte+Ticks übernommen, Rekorde bei 0,
Epsilon korrekt neu berechnet), Überschreib-Schutz pro Brett, ReLU/tanh
Round-Trip inkl. alter Checkpoints ohne `activation`-Feld, kompletter
frischer 17×15-Lauf mit den neuen Defaults, `watch_ai.py`-Champion-Suche.
Alle `.py`-Dateien im Projekt kompilieren fehlerfrei.

**Bewusst NICHT umgesetzt in dieser Runde** (Zeitgründe — nächste Session):
- **Phase 0.1** Post-Run-Report (Todesursachen nach Längen-Eimern usw.) —
  der wertvollste noch fehlende Baustein für datenbasierte Entscheidungen.
- **Phase 0.3 (Rest)**: die doppelte Bestätigungs-Prüfung vor
  Champion-Speicherung (nur die Episoden-Zahl wurde erhöht).
- **Phase 0.2, 0.4, 0.5**: Einsperr-Analyse, Meilenstein-Bibliothek, A/B-Runner.
- **Phase 1.2/1.3**: Dueling-Kopf, Lernraten-Treppe.
- **Phase 2 komplett**: Längen-Balance, Curriculum, Formung-Ausblendung +
  Sieg-Bonus, Neugier-Boden, Symmetrie-Verdopplung, wachsende Prüf-Notbremse.
- **Phase 3**: alle Performance-Optimierungen (Board-Zeichnen-Drossel etc.).
- **Phase 4**: CNN (wie geplant separat, erst nach Messung von 0-2).

**Nächster empfohlener Schritt:** Phase 0.1 (Post-Run-Report), weil jede
weitere Änderung (Phase 2) ohne die Längen-Eimer-Auswertung nur schwer zu
beurteilen ist.

---

# Teil A — Getroffene Entscheidungen

## A1. Brettgröße: 17×15 (offizielle Google-Snake-Maße) wird Standard

`grid_cols=17, grid_rows=15` (17 breit, 15 hoch). Begründung:
1. **Echte Bedingungen**: Das ist das Brett des bekanntesten Online-Snake.
   Ein dort trainierter Bot ist später direkt auf das echte Spiel
   übertragbar (SnakeAnalyzer-Anbindung).
2. **Ehrliche Schwierigkeit**: 17×15 = 255 Zellen — ungerade. Auf einem
   Brett mit ungerader Zellenzahl existiert beweisbar KEIN geschlossener
   Rundkurs durch alle Felder (Schachbrettfärbung: ein Rundkurs wechselt
   mit jedem Schritt die Farbe, braucht also gleich viele schwarze wie
   weiße Felder — bei 255 unmöglich). Der Bot kann sich also nie eine
   „sichere ewige Runde" antrainieren; 100% erfordert echte, flexible
   Raumplanung. Genau die Fähigkeit, die wir wollen.
3. Konsequenz: Auch die Curriculum-Bretter (A2) sind ungerade×ungerade,
   damit klein dieselbe Art Problem trainiert wie groß.

Alt-Bestand: Der bisherige 20×20-Champion (84.0) bleibt als Datei erhalten
und anschaubar; er kann sogar als Startpunkt für 17×15 dienen (B3), weil
die egozentrischen Wahrnehmungen brettgrößen-unabhängig sind.

## A2. Curriculum-Bretter: 9×9 → 13×11 → 17×15

Klein-Feld-Training (Endspiel-Übung, siehe C-Phase 2): 9×9 (81 Zellen)
und 13×11 (143 Zellen) — beide ungerade. Auf 9×9 erreicht der Bot „Feld
fast voll" hunderte Male pro Stunde statt einmal alle 50 Partien.

## A3. Config-Defaults

Die Hyperparameter-Defaults in `ai/dqn/config.py` sind bereits die
gemessene Bestkombi (heute gesetzt): perception=rich (Menü-Empfehlung:
rich_grid7), hidden=(256,128), num_games=16, lr=1e-3, gamma=0.97,
n_step=3, prioritized=False, fruit_count=3, eps_decay_steps=80k.
Durch diesen Plan ändern sich zusätzlich als neue Defaults:
- `grid_cols=17, grid_rows=15` (A1)
- `eval_episodes=20` statt 10 (C-0.3)
- `activation="relu"` neu (C-1.1)
Alles andere bleibt, bis A/B-Messungen etwas anderes belegen.

## A4. Leitplanke (unverändert gültig)

Kein Pathfinding, kein Flood-Fill, keine Heuristik als KI-EINGABE oder im
Handeln. Flood-Fill ausschließlich als Post-Mortem-TELEMETRIE (C-0.2).
Die „Sofort-Tod-Maskierung" (Abschnitt C-9) wird NICHT umgesetzt, solange
Luca nicht ausdrücklich zustimmt.

---

# Teil B — Bedienungsanleitung für Luca

## B1. Jetzt, vor der Umsetzung

Deinen laufenden 20×20-grid7-Lauf weiterfahren, solange die Prüfung
steigt (`P` drücken vor jedem Beenden!). Nachts:
`python train_dqn.py --headless 480 --weiter`.
NICHT starten: full_board (erst mit CNN sinnvoll), PER an, Lernrate 0.002.

## B2. Nach Umsetzung von Teil C, Phase S0+0 (Brett + Reports)

1. Frischer Lauf auf 17×15 (neuer Standard), Wahrnehmung rich_grid7,
   sonst Defaults. Das wird die neue Referenz-Messung.
   Option: statt frisch → Transfer vom 20×20-Champion (B3) — beides
   einmal messen, Report vergleichen, das bessere behalten.
2. Jeden Lauf beenden wie gewohnt — der Report entsteht automatisch in
   `logs/`. Diesen Report Claude zeigen (oder einfach sagen „lies den
   letzten Report") → nächster Schritt wird datenbasiert entschieden.

## B3. Klein→Groß-Wechsel (so läuft er ab, Schritt für Schritt)

Voraussetzung: C-Phase S0 umgesetzt. Funktioniert mit allen
egozentrischen Wahrnehmungen (simple, rich, rich_grid5/7/9) — NICHT mit
full_board (dort hängt die Netzgröße am Brett; erst mit CNN aus Phase 4).

1. Im Menü: Brettgröße „Klein (9×9)", Wahrnehmung rich_grid7, frisches
   Training starten. Ziel: Prüfung ≥ ~45 (das ist auf 81 Zellen schon
   >55% Füllung — Endspiel-Territorium).
2. `P` drücken, Esc, Fenster zu. (Champion landet in
   `models/dqn_champion_9x9.pt`.)
3. Neu starten: Brettgröße „Mittel (13×11)" + „Champion weitertrainieren:
   Ja". Das Menü zeigt an, dass ein Brett-Transfer stattfindet
   (Gewichte + Hyperparameter kommen mit, Brett und Rekorde starten neu).
   Trainieren bis Prüfung ≥ ~80 (>55% von 143).
4. Gleicher Schritt auf „Offiziell (17×15)". Ab hier ist es der Haupt-Bot.
5. Headless geht genauso: `python train_dqn.py --headless 480 --weiter`
   übernimmt das Brett des Champions; Wechsel des Bretts headless:
   `python train_dqn.py --headless 480 --weiter --brett 17x15`.

Wichtig zu wissen: Beim Brett-Wechsel werden Bestwerte/Prüfungs-Rekorde
zurückgesetzt (Scores verschiedener Bretter sind nicht vergleichbar —
je Brett gibt es eine eigene Champion-Datei). Die GEWICHTE (das Gelernte)
kommen vollständig mit.

---

# Teil C — Umsetzungs-Spezifikation für Sonnet 5

**Arbeitsregeln:** Schritte in dieser Reihenfolge. Nach jedem Schritt:
Smoke-Test (unten je Schritt definiert) + `python -m py_compile` auf alle
geänderten Dateien. Kommentarstil des Projekts beibehalten (deutsch,
erklärend, Leitplanken-Hinweise). Nichts committen ohne Lucas OK.
Bestehende Checkpoints dürfen NIE stillschweigend falsch geladen werden —
im Zweifel klare Fehlermeldung.

## Phase S0 — Brettgröße 17×15 + Brett-Infrastruktur (ZUERST)

### S0.1 Neue Defaults
- `ai/dqn/config.py`: `grid_cols: int = 17`, `grid_rows: int = 15`,
  Kommentar: Google-Snake-Maße, ungerade Zellenzahl ⇒ kein sicherer
  Rundkurs möglich (siehe TRAININGSPLAN.md A1).
- `game/config.py` (Mensch-Spiel) NICHT ändern — das Menschen-Spiel
  bleibt 20×20; nur die KI-Trainingsumgebung wechselt.

### S0.2 full_board dynamisch machen (behebt latenten Bug)
`ai/perception.py`: `perceive_full_board`/`FULL_BOARD_INPUT_SIZE`/Labels
sind auf `GRID_COLS/GRID_ROWS` (20×20) festverdrahtet — bei 17×15 stimmt
`input_size` (411) nicht mehr mit dem echten Vektor (11+255=266) überein.
Umbau analog `make_rich_grid_perception`:
- `make_full_board_perception(cols, rows) -> (fn, size, labels)`.
- Registry: `PERCEPTIONS["full_board"]` bekommt einen Factory-Marker;
  `get_perception(name, cols=None, rows=None)` erweitert: für
  brettabhängige Wahrnehmungen müssen cols/rows übergeben werden (sonst
  klare Fehlermeldung). Trainer ruft `get_perception(cfg.perception,
  cfg.grid_cols, cfg.grid_rows)` auf; `watch_ai.py` nutzt die Brettmaße
  aus dem Checkpoint. Egozentrische Wahrnehmungen ignorieren cols/rows.
- Der `GRID_COLS/GRID_ROWS`-Import aus game/config.py entfällt.
Abnahme: Testskript baut Spiele 9×9/13×11/17×15/20×20, prüft für JEDE
registrierte Wahrnehmung `fn(game).shape == (size,)` und
`len(labels) == size`.

### S0.3 Champion-Datei pro Brett
`ai/dqn/trainer.py`:
- Neue Funktion `champion_path(cols, rows) -> str`:
  `models/dqn_champion_{cols}x{rows}.pt`. Übergangsregel: existiert für
  (20,20) keine neue Datei, aber die alte `models/dqn_champion.pt`, wird
  die alte gelesen (Legacy), geschrieben wird immer ins neue Schema.
- Trainer benutzt durchgehend `champion_path(cfg.grid_cols,
  cfg.grid_rows)` (Speichern, Resume-Default, eval_best-Untergrenze).
  Die eval_best-Untergrenze (Überschreib-Schutz) gilt nur noch für die
  Datei DESSELBEN Bretts — Scores anderer Bretter sind unvergleichbar.
- `watch_ai.py dqn`: lädt per Default den 17×15-Champion; existiert er
  nicht, den neuesten vorhandenen `dqn_champion_*.pt` (mtime) und sagt
  dazu, welches Brett gezeigt wird. `load_champion_config()` bekommt den
  Pfad als Pflichtargument statt Default.
Abnahme: Smoke-Test — Training 9×9 speichert `dqn_champion_9x9.pt`;
Training 17×15 findet/überschreibt den 9×9-Champion NICHT.

### S0.4 Brett-Transfer beim Weitertrainieren
- Menü (`dashboard/dqn_view.py`): neue Zeile „Brettgröße" mit Presets
  `[("Klein (9×9)", (9,9)), ("Mittel (13×11)", (13,11)),
  ("Offiziell (17×15)", (17,15)), ("Klassisch (20×20)", (20,20))]`,
  in `_sync_menu_from_config` und `_start_training` verdrahtet.
- „Champion weitertrainieren: Ja" + abweichende Brettgröße = Transfer:
  - Nur zulässig, wenn `input_size` von Wahrnehmung des Checkpoints ==
    aktueller (bei egozentrischen immer erfüllt); sonst die bestehende
    klare Fehlermeldung.
  - Es gilt: Gewichte + alle Hyperparameter aus `full_config` übernehmen,
    ABER `grid_cols/grid_rows` (und damit Brett) aus der Menü-Auswahl;
    `eval_best=0`, `eval_max=0`, `best_score=0` (neues Brett, neue
    Rekorde); `total_steps/total_episodes` übernehmen (damit epsilon
    unten bleibt — der Bot ist ja erfahren).
  - Log-Zeile + Menü-Hinweis: „Brett-Transfer 9×9 → 13×11".
- CLI (`train_dqn.py`): neues Argument `--brett CxR` (z.B. `--brett
  17x15`), nur zusammen mit `--weiter` sinnvoll; ohne `--brett` gilt das
  Brett des Champions. `--weiter` lädt den Champion passend zur
  Brettgröße der aktuellen Config (champion_path).
Abnahme: Save→Transfer-Test: 500 Ticks auf 9×9, Champion erzwingen
(`run_evaluation`), Trainer neu mit (13,11)+resume → `input_size` gleich,
`eval_best==0`, epsilon < 0.5, Spiele laufen auf 13×11.

## Phase 0 — Messfundament

### 0.1 Post-Run-Report
Neue Datei `ai/dqn/report.py`:
- Trainer sammelt ab sofort pro beendeter Episode:
  `(länge_bei_tod, ursache, kopf_x, kopf_y, score, züge)` in eine Liste
  (RAM unkritisch; bei >200k Episoden älteste halbieren wie score_curve).
- `write_report(trainer, pfad_basis)` erzeugt
  `logs/dqn-<runid>-report.json` UND lesbares `-report.md`:
  - Config + Diff zu `DQNConfig()`-Defaults
  - Laufzeit, Ticks, Episoden, Ø Züge/s
  - Prüfungs-Verlauf (Liste), Bestwert, Champion-Datei + ob neu
  - **Todesursachen je Längen-Eimer** (0–9, 10–19, …, ≥90): absolute
    Zahlen + Prozent je Eimer — Kernstück
  - Züge/Frucht je Längen-Eimer; Score-Histogramm (Eimer 10);
    Todes-Positions-Raster (cols×rows-Zählmatrix)
  - Q-Kalibrierung: bei jeder Prüfung zusätzlich mittleren
    Start-Q-Wert der Prüfpartien vs. tatsächlich erzielte
    (diskontierte) Belohnung protokollieren → ins Report übernehmen
  - `diagnosen`: Liste automatischer Textbefunde, mindestens:
    Selbstkollision >60% in einem Eimer ≥30; Prüfung stagniert (letzte 5
    Prüfungen alle < Bestwert − 5%); Loss-Trend steigend über letztes
    Drittel; Einsperr-Quote (0.2) > 50%.
- Aufrufstellen: `run_headless` (finally-Block, auch bei Strg+C),
  `DQNDashboard` bei Esc→Menü und bei QUIT (nur wenn ein Trainer läuft;
  doppeltes Schreiben pro Lauf ok — Datei wird überschrieben).
Abnahme: Kurzlauf headless 0.2 min → beide Dateien existieren, JSON
parsebar, Eimer-Summen == Gesamt-Tode.

### 0.2 Einsperr-Analyse (NUR Telemetrie — Leitplanken-Kommentar Pflicht)
In `report.py` (nicht in perception/agent!): bei Todesursache „self" nach
dem Tod Flood-Fill vom Kopf über freie Zellen → `erreichbare_zellen`,
`frei_gesamt`, Quote. Pro Episode mitloggen, im Report je Längen-Eimer
mitteln („eingesperrt" := Quote < 25%). Der Wert darf NIRGENDS in
Wahrnehmung/Belohnung/Aktionswahl auftauchen.
Abnahme: konstruierter Spielzustand mit eingesperrtem Kopf → Quote < 25%.

### 0.3 Ehrlicher Champion
`ai/dqn/config.py`: `eval_episodes=20`. `trainer.run_evaluation`: wenn
`mean > eval_best`, VOR dem Speichern Bestätigung: weitere 20 Partien,
Champion nur wenn Gesamtmittel (40 Partien) > eval_best; Bestätigungswert
zählt als der neue eval_best. min/median/max jeder Prüfung ins CSV (drei
neue Spalten) und in die Report-Historie.
Abnahme: bestehender Save→Resume-Smoke-Test weiter grün; CSV hat neue
Spaltenzahl (Kopfzeile angepasst).

### 0.4 Meilenstein-Bibliothek
Nach jedem neuen Champion: Füllgrad = (eval_best + 3) / (cols*rows).
Beim erstmaligen Überschreiten von 20/25/30/40/50%:
Checkpoint-Kopie nach `models/milestones/dqn_{cols}x{rows}_fill{pct}.pt`.
Abnahme: Kurztraining mit künstlich kleinem Brett erreicht 20% → Datei da.

### 0.5 A/B-Runner
Neues Skript `run_experiments.py` (Repo-Wurzel):
- Liest `experiments.json`: Liste von `{name, minuten, seeds:[..],
  overrides:{configfeld: wert}}`.
- Führt sie nacheinander headless aus (eigener Prozess je Lauf via
  `subprocess`, damit Torch-Threads sauber neu starten; `--seed` als
  neues CLI-Argument in train_dqn.py durchreichen).
- Schreibt `logs/experiments-<zeit>-summary.md`: Tabelle Name/Seed →
  finale Prüfung, beste Prüfung, Züge/s, Report-Pfad.
Abnahme: Beispiel-experiments.json mit 2 Mini-Läufen (0.2 min) läuft
durch, Summary-Tabelle vollständig.

## Phase 1 — Netz-Grundlagen

### 1.1 ReLU (Erwartung: größter Einzeleffekt)
- `ai/torch_bridge.py`: `SnakeNet(hidden, input_size,
  activation="tanh")`; forward nutzt das Feld. Default BLEIBT "tanh"
  (Rückwärtskompatibilität + Neuroevolution unverändert).
- `ai/dqn/config.py`: neues Feld `activation: str = "relu"`. Agent baut
  beide Netze damit. Checkpoint-Payload + Ladepfade (`_resume`,
  watch_ai.load_champion, torch_bridge-Leser) speichern/lesen
  `activation`; fehlt das Feld im Checkpoint → "tanh" annehmen.
  `_resume` wirft klare Fehlermeldung bei Aktivierungs-Mismatch (wie bei
  Netzgröße).
- A/B (über 0.5): relu vs tanh, 2 Seeds, gleiches Budget, 17×15.
Abnahme: alter tanh-Checkpoint lädt und spielt unverändert; neuer Lauf
speichert activation="relu" im Checkpoint.

### 1.2 Dueling-Kopf
- `torch_bridge.SnakeNet`: Parameter `dueling: bool = False`. Wenn True:
  nach den Hidden-Schichten zwei Köpfe — Value (Linear→1) und Advantage
  (Linear→3); `Q = V + A − A.mean(dim=1, keepdim=True)`.
- Config-Feld `dueling: bool = True` (DQN-Default an, Checkpoint-Feld +
  Mismatch-Fehler analog 1.1; Neuroevolution nutzt weiter dueling=False).
- A/B gegen dueling=False.
Abnahme: Forward-Shape (batch,3); Checkpoint-Roundtrip.

### 1.3 Lernraten-Treppe
Config: `lr_meilensteine: tuple = ((90, 3e-4), (130, 1e-4))` —
bei Prüfungs-Bestwert ≥ Schwelle wird die Optimizer-LR EINMALIG gesenkt
(im Trainer nach run_evaluation; `optimizer.param_groups[0]["lr"]`).
Log-Zeile + Report-Eintrag. (Schwellen gelten für 17×15; im Report
dokumentieren.)
Abnahme: Unit-ähnlicher Test mit gemocktem eval_best.

## Phase 2 — Endspiel-Training

### 2.1 Längen-balanciertes Lernen (+ Puffer 300k)
- Puffer speichert zusätzlich je Eintrag die Schlangenlänge (uint16-Array
  parallel zu den bestehenden Arrays, auch im n-Schritt-Pfad).
- Config: `buffer_size=300_000`, `balance_min_laenge: int = 30`,
  `balance_anteil: float = 0.3`.
- `sample(batch)`: `k = int(batch*anteil)` Indizes gleichverteilt aus
  Einträgen mit Länge ≥ min_laenge (zweite Indexliste inkrementell
  pflegen, Ringpuffer-Überschreiben beachten!); Rest gleichverteilt aus
  allen. Gibt es < k lange Einträge → so viele wie da sind, Rest normal
  (kein Absturz, kein Bias am Anfang).
- Gilt für den Standard-Puffer; PER-Pfad unverändert lassen (Retest 2.8
  entscheidet später).
Abnahme: synthetischer Puffer, Anteil langer Samples ≈ 30% (±5%);
NumPy-Aliasing-Regressionstest (siehe Memory numpy-zeilen-aliasing-bug:
Kopien, keine Views!).

### 2.2 Endspiel-Curriculum (ERST NACH LUCAS OK)
- Bei jeder Prüfung: von der besten Prüfpartie Snapshots bei Länge
  40/50/60 (deepcopy von snake/fruits/direction/steps_since_fruit) in
  eine Datei `models/startstellungen_{cols}x{rows}.pkl` (Ring, max 200).
- Config `curriculum_anteil: float = 0.25`: beim Episodenstart (reset in
  `_finish_episode` und Initialbelegung) mit dieser Wahrscheinlichkeit
  eine zufällige gespeicherte Stellung laden statt Länge 3.
  Prüfungen (run_evaluation) IMMER von Länge 3 — der Maßstab bleibt rein.
- KI erhält keinerlei Zusatzinfo; nur die Startbedingung ändert sich.
Abnahme: mit Anteil 1.0 starten alle Spiele lang; Prüfung startet kurz.

### 2.3 Klein-Feld-Curriculum
Kein neuer Code nötig — ist S0.4 (Brett-Transfer) + Ablauf B3.

### 2.4 Formung ausblenden + Sieg-Bonus
- Config: `formung_aus_ab: float = 50.0`, `formung_null_ab: float =
  80.0`, `reward_win: float = 100.0`. (Schwellen beziehen sich auf
  eval_best des aktuellen Bretts; für 9×9 skaliert der Trainer sie mit
  Faktor (Zellen/255).)
- Trainer berechnet nach jeder Prüfung
  `formung_faktor = clamp((null_ab − eval_best)/(null_ab − aus_ab), 0, 1)`
  und übergibt ihn an compute_reward; dort multipliziert er NUR
  closer/farther. `result.won` → zusätzlich reward_win.
Abnahme: Faktor-Formel-Test (eval_best 40→1.0, 65→0.5, 90→0.0);
won-Übergang liefert reward ≈ 110.

### 2.5 Neugier-Boden senken
Config `eps_end_spaet: float = 0.005`, `eps_spaet_ab: float = 70.0`
(brett-skaliert wie 2.4): ab Schwelle gilt eps_end_spaet in der
Epsilon-Formel (Trainer, beide Stellen: step() und Resume-Neuberechnung).
Abnahme: Formel-Test.

### 2.6 Symmetrie-Verdopplung
- `ai/perception.py`: je Wahrnehmung eine Spiegel-Permutation +
  Vorzeichen-Maske definieren (`MIRROR_MAPS[name] -> (perm, sign)`), die
  den Vektor der links-rechts-gespiegelten Welt erzeugt:
  gespiegelt = vektor[perm] * sign. Aktionen: STRAIGHT bleibt,
  LEFT↔RIGHT tauschen (aus dem Action-Enum ableiten, nicht hart kodieren).
- **Abnahmekriterium ist der Test, nicht die Tabelle**: Für jede
  Wahrnehmung einen Spiegel-Spielzustand konstruieren (Schlange +
  Früchte an x gespiegelt, Richtung gespiegelt) und prüfen:
  `mirror(perceive(game)) == perceive(mirrored_game)` (atol 1e-6).
  full_board: zusätzlich Brett-Spalten spiegeln.
- `memory.sample()`: mit 50% Wahrscheinlichkeit je gezogenem Eintrag
  Zustand+Folgezustand spiegeln und Aktion tauschen (Kopien! kein
  In-Place auf Pufferzeilen). Config `spiegel_lernen: bool = True`.
Abnahme: der Spiegel-Test oben; Trainings-Smoke-Test 500 Ticks ohne
Loss-Explosion.

### 2.7 gamma-A/B (0.97 vs 0.99) — über den A/B-Runner, NACH 2.1.
### 2.8 PER-Retest — über den A/B-Runner, NACH 2.1. Kein Code.

### 2.9 Prüf-Notbremse mitwachsen lassen
`eval_max_steps` dynamisch: `max(cfg.eval_max_steps,
50 * (eval_best + 20))` zur Laufzeit in run_evaluation. Damit deckelt die
4000er-Bremse nie gute Endspiele (S8). Verhungern-Limit bleibt wie es ist.
Abnahme: eval_best=150 → Limit ≥ 8500.

## Phase 3 — Performance (ohne Funktionsverlust)

### 3.1 Spielfeld-Zeichnung drosseln
`dqn_view.py`: Miniatur-Felder nur alle 100 ms neu zeichnen (letztes Bild
als Surface cachen und blitten); Kacheln/Kurven/Header weiterhin jedes
Frame. Abnahme: optisch identisch, Züge/s im Fenster-Modus auf Lucas
Windows-PC messbar höher (er berichtet ~4.000 vorher).

### 3.2 Text-Cache Dashboard
Render-Cache dict[(text, font, farbe)] → Surface, Größe begrenzen (~500
Einträge, dann leeren). Wie im Spiel-Renderer.

### 3.3 Wahrnehmung vektorisieren
Pro Spiel ein `np.int8`-Brett (0 leer, 1 Körper, 2 Frucht), gepflegt über
einen kleinen Hook im Trainer nach jedem Zug (Kopf setzen, ggf. Schwanz
löschen, Frucht-Respawn) — NICHT im Spiel selbst (Engine bleibt
KI-frei). grid-Ausschnitt = Slice + `np.rot90` je Blickrichtung;
full_board = Brett flach + Skalare. Rays optional später.
Abnahme: für 1000 zufällige Zustände identische Vektoren wie die alte
Implementierung (atol 0), Durchsatz-Vergleich im Report festhalten.

### 3.4 train_every/batch-A/B — Runner-Experiment, kein Code:
(train_every=2, batch=512) vs Default, Maßstab Prüfung nach gleicher
WANDUHR-Zeit.

### 3.5 GPU-Check Windows — Luca führt aus:
`python -c "import torch; print(torch.cuda.is_available())"` → Ergebnis
in den nächsten Report/Chat.

## Phase 4 — Der große Sprung (separater Auftrag, erst nach Messung 0–2)

4.1 CNN (`network:"cnn"`, Kanäle Körper/Kopf/Frucht roh, 2–3 Conv →
concat mit Skalaren → FC → 3 Q bzw. Dueling; brettgrößen-tolerant via
adaptivem Pooling → verbindet sich mit Klein-Feld-Curriculum).
4.2 Beobachtungs-Stapel (letzte k=3 Vektoren, Config `frame_stack`).
4.3 EMA-Gewichte für Prüfung/Champion (tau=0.999).
4.4 Noisy Nets / C51 nur falls 4.1–4.3 nicht reichen.
Für Phase 4 vor Umsetzung eine eigene Detail-Spez schreiben (Netz-I/O,
Checkpoint-Format, watch_ai) und von Luca freigeben lassen.

## C-9. NICHT umsetzen ohne ausdrückliches OK
- Sofort-Tod-Maskierung (auch nicht „nur für Zufallszüge").
- Endspiel-Curriculum 2.2 (wartet auf Lucas OK).
- Automatisches Hyperparameter-Tuning während des Laufs.

## C-10. Messprotokoll (gilt für JEDEN A/B)
Gleiches Wanduhr-Budget, ≥2 Seeds je Arm, 17×15, Bericht = Prüfungs-
Bestwert + Endwert + Todes-Eimer-Vergleich aus den Reports. Sieger wird
nur Default, wenn er in beiden Seeds vorn liegt; sonst dokumentieren und
Default behalten. Immer nur EIN Unterschied pro Vergleich.
