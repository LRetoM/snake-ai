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

## UMSETZUNGSSTAND (2026-07-23, von Sonnet 5 — 2. Runde)

### Runde 5 (2026-07-24, Mess-Fundament-Generalueberholung — nach Lucas
"wir muessen uns auf alle Werte verlassen koennen")

**Ausloeser**: Der 45-Min-Lauf `dqn-20260724-152712` (rich_grid7, frisch,
Curriculum an, nach Memory-Fix stabil 11.177 Zuege/s ueber den GANZEN Lauf)
zeigte drei Reporting-Fehler, die Messungen unbrauchbar machten:
1. Die Champion-Schutzschwelle (87.25, vom rich_grid9-Lauf!) speiste auch
   Meilensteine + Diagnosen: der frische Lauf lief ab Tick 1 mit
   abgeschalteter Formung + abgesenktem Neugier-Boden und der Report
   meldete "Plateau" gegen einen fremden Bestwert.
2. Q-Kalibrierungs-Text behauptete immer "ueberschaetzt sich", auch bei
   negativer Differenz (und ignorierte den systematischen gamma-Abzins-
   Anteil).
3. Curriculum-Starts vermischten sich unsichtbar mit natuerlichen Partien
   in der Laengen-Statistik (23.803 Episoden im Eimer 60-69 vs. 1.763 in
   Runde 3 — groesstenteils Konstruktions-Artefakt, kein Vergleich moeglich).

**Umgesetzt** (alles verifiziert: py_compile, Unit-Checks, End-to-End-
Smoke-Lauf mit Report-Inspektion):
- `trainer.py`: **eval_best_run** (Niveau DIESES Laufs) von **eval_best**
  (Schutzschwelle der Datei) getrennt. Formung/Neugier-Boden/LR-Treppe/
  Diagnosen haengen jetzt am Lauf-Niveau; die Schwelle schuetzt NUR noch
  die Champion-Datei. Bei --weiter auf demselben Brett erbt eval_best_run
  das Champion-Niveau (der Bot IST ja so gut). `champion_floor_info`
  protokolliert Herkunft (Wert + Wahrnehmung) der Schwelle.
- **Runbest-Checkpoint** `models/dqn_runbest_<cols>x<rows>.pt`: bester
  Stand jedes Laufs wird IMMER gesichert — vorher warf ein Lauf, der die
  fremde Schwelle nicht knackte, sein gesamtes Ergebnis weg.
- **Pruefungs-Historie** (`eval_history`): je Pruefung min/median/max
  (Streuung!), Zuege/s seit letzter Pruefung (haette den memory.py-
  Einbruch sofort gemeldet), Loss, Start-Q, epsilon. Als Tabelle im
  Report, komplett im JSON, min/median/max auch als neue CSV-Spalten.
- **Episode-Log + Report trennen Curriculum-Starts von natuerlichen
  Partien** (Flag je Episode; getrennte Todesursachen-Tabellen; Score-
  Histogramm nur natuerlich; Zuege/Frucht bei Curriculum ausgeblendet,
  weil der geerbte Start-Score die Zahl sinnlos macht).
- **Diagnosen ueberarbeitet**: Plateau gegen Lauf-Bestwert; Loss NORMIERT
  auf die Q-Skala (wachsende Q-Werte ziehen den absoluten Loss physikalisch
  mit hoch — erst ein steigendes Verhaeltnis ist ein Signal); NEU:
  Geschwindigkeits-Diagnose (Drittel-Vergleich der Zuege/s); NEU: Hinweis,
  wenn die Schutzschwelle von einer anderen Wahrnehmung stammt.
  Q-Kalibrierung jetzt richtungsbewusst mit Erklaerung des Abzins-Anteils.
- **Headless-Ausgabe** zeigt `Lauf-best` statt der fremden Schwelle (die
  stand dort den ganzen Tag als irrefuehrendes "best 87.2").
- **A/B-CLI**: `--seed N`, `--curriculum ANTEIL`, `--pfadfokus ANTEIL` —
  ein Unterschied pro Vergleich, ohne config.py anzufassen (C-10).
- **NEU: Pfad-Fokus-Regler** (Lucas Idee, "erst sicher, dann schnell",
  2026-07-24 auf EINEN Regler vereinheitlicht): `cfg.pfad_fokus` (0.0..1.0)
  mischt LINEAR zwischen reinem Fruchtsammeln (0.0, heutiges Verhalten)
  und reinem Ueberleben (1.0, Frucht bringt dann NULL Belohnung — ob
  gefressen oder nicht ist egal, nur noch der Ueberlebens-Bonus pro Zug
  zaehlt). Verhungern (starve_limit) zwingt trotzdem zum Fressen — Frucht
  wird so zum MITTEL fuers Ueberleben statt zum Selbstzweck. Der
  tatsaechlich wirksame Wert (`MultiGameTrainer.pfad_fokus_aktuell`)
  blendet wie die Formung deterministisch Richtung 0 aus, sobald das
  Lauf-Niveau steigt (Schwellen `pfad_fokus_aus_ab`/`_null_ab`, 80/120
  brett-skaliert) — der Bot hat dann einen sicheren Weg gefunden, jetzt
  wird er schrittweise wieder aufs Fruchtsammeln (= Tempo) getrimmt.
  Getestet: 0.0 exakt altes Verhalten, 1.0 Frucht=0/nur Bonus, Ausblenden
  bei steigendem Lauf-Niveau korrekt. Code-Standard AUS (`pfad_fokus=0.0`)
  bis eine A/B-Messung ihn belegt.
- Ausserdem aus derselben Session: `memory.py` Set→Array-Fix (Zuege/s
  stabil statt 11k→3k-Einbruch, 31.8x schnellere Batch-Ziehung im Test).

**Messstand rich_grid7 + Curriculum** (Lauf 152712, 45 Min, ~110k
Episoden): Lauf-Bestwert 63.8, hohe Laengen-Eimer unveraendert ~71-79%
Selbstkollision — ABER konfundiert (Runde 3 war rich_grid9 + fruit_count 10
+ eps_decay 150k) und durch die faelschlich abgeschaltete Formung verzerrt.
Erst die naechsten Laeufe mit dem reparierten Fundament sind belastbar.

**Nachtrag Runde 5b** (noch selber Tag, Lucas Live-Test im Fenster deckte
zwei weitere Probleme auf):
- **Cross-Run-Kontamination beim Curriculum** (derselbe Fehlertyp wie die
  Champion-Schutzschwelle, nur an zweiter Stelle): ein frischer Lauf lud
  bisher beim Start blind ALLE gespeicherten Stellungen von der Platte —
  auch von einem voellig anderen frueheren Lauf (andere Wahrnehmung, ganz
  anderer Trainingsstand). Sichtbarer Effekt: ein gerade erst gestartetes
  Netz bekam ab Episode 1 Laenge-40+-Stellungen eines fremden, viel
  staerkeren Bots vorgesetzt — sah im Fenster kaputt aus ("ploetzlich extrem
  lange Schlange") und war es konzeptionell auch, weil das Curriculum ja
  gerade die EIGENEN erreichten Grenzen widerspiegeln soll (Lucas: "macht
  natuerlich nur Sinn vom eigenen Run"). Fix: `curriculum_snapshots` wird
  nur noch geladen, wenn dieser Lauf tatsaechlich denselben Bot fortsetzt
  (`--weiter`/Champion-Weitertraining) — ein frischer Lauf startet immer
  mit leerem Vorrat und baut ihn nur aus seinen EIGENEN Pruefungen auf.
- **Trap-Erkennung** (Lucas Idee): manche gespeicherte Stellung koennte in
  Wirklichkeit schon unrettbar sein (die Pruefpartie war beim Einsammeln
  zwar noch am Leben, aber vielleicht schon in einer Falle ohne Ausweg).
  Jetzt wird je Stellung mitgezaehlt, wie oft jeder der 3 moeglichen
  ERSTEN Zuege (geradeaus/links/rechts) versucht wurde und wie oft das
  schnell (≤15 Zuege) in Wand/Selbst-Tod endete. Eine Stellung wird erst
  entfernt, wenn ALLE 3 Pfade mindestens 5x versucht wurden UND jeder davon
  zu ≥90% schnell gestorben ist — ein einzelner schlechter Pfad reicht
  nicht (koennte ueber einen anderen ersten Zug loesbar sein). Reine
  Trainings-Buchhaltung (Ringpuffer-Pflege), keine Info an die KI, keine
  Strategie. Entfernungen landen im Meilenstein-Log und im Report
  ("Curriculum: ... X als aussichtslos entfernt").
- **Menue erweitert**: Endspiel-Curriculum jetzt bis 100% waehlbar (0/10/25/
  40/60/80/100%), nicht mehr bei 40% gedeckelt — Lucas Wunsch, das selbst
  ausprobieren zu koennen.
- Getestet: frischer Lauf startet mit 0 geladenen Stellungen (auch wenn
  die Datei voll ist); Trap-Entfernung greift nur nach allen 3 Pfaden UND
  nur bei echter Aussichtslosigkeit (Gegentest: bleibt erhalten, wenn ein
  Pfad auch nur einmal eine echte Chance zeigte oder wenn ein Pfad noch gar
  nicht getestet wurde).

**Nachtrag Runde 5c** (noch selber Tag: Pfad-Fokus-Leck + Auto-Tuner):
- **Pfad-Fokus-Leck gefixt** (Lucas Live-Beobachtung: "selbst bei 100%
  wird noch Fruechte gesammelt"): die Naeher/Weiter-Formung lief beim
  Pfad-Fokus ungebremst weiter — bei 100% Fokus war die Frucht zwar 0
  wert, aber der WEGWEISER dorthin (+0.1/-0.12 je Zug) staerker als der
  Ueberlebens-Bonus (0.05). Jetzt skaliert die Formung mit
  `(1 - pfad_fokus_aktuell)` mit runter. Getestet ueber alle
  fokus×formung-Kombinationen: bei 100% Fokus ist die Belohnung komplett
  richtungs- und frucht-blind (Frucht-, Naeher-, Weg-, Neutral-Zug alle
  exakt gleich), einziger Frucht-Anreiz bleibt indirekt die
  Verhungern-Regel — genau wie gewollt.
- **Curriculum-Persistenz nur noch fuer die Champion-Linie**: frische
  Laeufe schreiben ihren Stellungs-Vorrat nicht mehr auf die Platte
  (haetten sonst den Vorrat des Champions ueberschrieben — dieselbe
  Cross-Run-Kontamination wie beim Laden, nur rueckwaerts; besonders
  kritisch, sobald der Auto-Tuner dutzende frische Laeufe macht).
- **`--override FELD=WERT`** in train_dqn.py (mehrfach angebbar): JEDES
  DQNConfig-Feld laesst sich fuer einen einzelnen Headless-Lauf setzen;
  unbekannte Feldnamen sind ein harter Fehler. Grundlage fuer den Tuner.
- **NEU: `auto_tuner.py`** (Lucas Auftrag "vollautonome Config-Suche"):
  laeuft unbeaufsichtigt (z.B. ueber Nacht/ganzen Tag), misst erst die
  Basis (Standard + rich_grid7) mit festen Seeds und gleichem
  Wanduhr-Budget je Lauf (Standard 12 Min, 2 Seeds), aendert dann je
  Kandidat GENAU EINE Einstellung aus einem definierten Suchraum (21
  Parameter: lr, gamma, n_step, batch, hidden, num_games, fruit_count,
  eps-*, train_every, target_update, perception, curriculum_anteil,
  pfad_fokus(+bonus), balance_anteil, spiegel_lernen, prioritized,
  activation, reward_death/step), vergleicht gegen die Bestmarke
  (Annahme nur bei >2% Verbesserung), verwirft sonst. Verworfene Werte
  werden mit 20% Wahrscheinlichkeit bis zu 3x ERNEUT getestet (gegen
  Pech-Laeufe). Alle 8 Kandidaten wird die Bestmarke frisch nachgemessen
  (gegen Waerme-Drift des Macs). Bewusst SEQUENZIELL statt 5 parallel:
  parallele Laeufe teilen sich CPU/Waermebudget → Wanduhr-Scores waeren
  unvergleichbar, dazu Datei-Kollisionen. Protokoll je Tuner-Lauf in
  `logs/autotuner-<zeit>/`: laeufe.csv (jeder einzelne Trainingslauf mit
  Config+Score+Report-Verweis), protokoll.md (jede Entscheidung),
  zustand.json (persistent, per `--fortsetzen` wiederaufnehmbar),
  abschlussbericht.md (beste Config + Verlauf + "Gelerntes je Parameter"),
  plus Modell-Sicherung vor dem Start. Stoppt bei Konvergenz (30
  Verwerfungen in Folge), nach `--max-stunden` (Standard 24) oder per
  Strg+C — Abschlussbericht kommt in allen Faellen. End-to-end getestet
  inkl. Fortsetzen und Retest-Mechanik.
  Start: `python auto_tuner.py` (Optionen: `--minuten`, `--seeds`,
  `--max-stunden`, `--konvergenz`, `--marge`, `--fortsetzen`).

**Naechster Schritt (Empfehlung)**: Zwei saubere A/B-Paare mit je 2 Seeds,
gleiches Zeitbudget, identische Basis (rich_grid7, Standard-Defaults):
1. `--curriculum 0` vs. `--curriculum 0.25` — wirkt das Curriculum?
2. `--pfadfokus 0` vs. `--pfadfokus 0.7` (o.ae.) — bringt der Pfad-Fokus-
   Regler die besseren Pfade?
Danach datenbasiert entscheiden; bleibt Selbstkollision in den hohen
Eimern trotz allem hart, ist die Wahrnehmung die Grenze → Phase 4 (CNN).

### Runde 1 (Brett-Infrastruktur + ReLU)
- **S0.1-S0.5**: Neue Defaults `grid_cols=17, grid_rows=15`; `full_board`
  brettgrößen-dynamisch (`make_full_board_perception`, `get_perception(name,
  cols, rows)`); Champion-Datei pro Brettgröße (`champion_path()`/
  `resolve_champion_path()` in `ai/dqn/trainer.py`, Schema
  `dqn_champion_<cols>x<rows>.pt`, alte `dqn_champion.pt` als Legacy-Lesepfad);
  Menü-Zeile "Brettgröße" + Brett-Transfer beim Weitertrainieren; CLI
  `--brett SPALTENxZEILEN` (nur `--headless`).
- **Phase 1.1**: `activation`-Feld (relu/tanh), DQN-Default `relu`,
  Neuroevolution bleibt `tanh`.
- **Bugfix (nach Rückmeldung)**: Menü-Zeile "Champion weitertrainieren"
  zeigte nach einem Brett-Wechsel fälschlich "kein Champion an", obwohl der
  Transfer intern korrekt vorbereitet war (reine Anzeige-Logik, nicht die
  eigentliche Resume-Funktion) — gefixt, zeigt jetzt "Ja (Transfer von AxB)".
- **Bugfix**: `_eval_games` war auf `min(eval_episodes, 16)` gedeckelt und
  wuchs bei nachträglicher Änderung von `cfg.eval_episodes` (z.B. die
  Abschluss-Prüfung in `train_dqn.py`) nie nach — lief still mit weniger
  Partien als eingestellt. Jetzt behoben, `eval_episodes` 10→20.

### Runde 3 (Messung, 2026-07-24 — Report `dqn-20260724-124234`)
33.6 Min, 60.575 Episoden, Brett 17×15. **Achtung, kein sauberer Vergleich
zur Runde-1-Basis**: dieser Lauf wich vom empfohlenen Standard ab
(`perception=rich_grid9` statt `rich_grid7`, `fruit_count=10` statt 3,
`eps_decay_steps=150000` statt 80000 — alles Lucas manuelle Overrides,
nicht neue Defaults).
- **Neuer Champion: Prüfung 87.25** (≈ (87.25+3)/255 ≈ 35% Feldfüllung —
  Fortschritt gegenüber der 19.2%-Basis, deckt sich mit Lucas eigener
  ~33%-Schätzung).
- **Kernbefund, der Phase 2.2 direkt stützt**: Selbstkollision steigt
  MONOTON mit der Schlangenlänge — 58.3% (Länge 0-9) → 63.0% → 53.2% →
  51.1% → 58.2% → 64.9% → 70.6% → 73.3% → 77.8% → **81.0% (Länge 90+)**.
  Wand-Tod sinkt spiegelbildlich (41.7% → 19.0%). Die Basis-Messung aus
  Runde 1 hatte nur eine einzige Gesamtzahl (52%); jetzt zeigt sich die
  Zunahme mit der Länge explizit, genau das erwartete "Einsperr"-Muster,
  je länger desto häufiger.
- Q-Kalibrierung sauber (Q 38.12 vs. tatsächlich 38.15 — kein
  Selbstüberschätzungs-Problem).
- Auto-Diagnose meldet **Plateau** (letzte 5 Prüfungen alle < 95% von
  87.25) — deckt sich mit Lucas eigener Live-Beobachtung "schon wieder
  auf einem Plateau angelangt".
- Auto-Diagnose meldet **steigenden Loss** im letzten Drittel (2.60 →
  4.61). Grund vermutlich: `lr_meilensteine` ((90, 3e-4), (130, 1e-4))
  hat bei eval_best 87.25 noch nicht gegriffen (Schwelle 90 knapp
  verfehlt) — nächster Lauf sollte das von selbst lösen, sobald die
  Prüfung die 90 überschreitet; falls nicht, Lernrate ist ein Kandidat.
- **Konsequenz**: Phase 2.2 (Endspiel-Curriculum) wird jetzt umgesetzt
  (Lucas OK am 2026-07-24) — der Report liefert die Begründung direkt:
  das Problem sitzt eindeutig im langen Spiel, nicht in Wand-Kollisionen
  oder Grundlagen.

### Runde 2 (Phase 2 + Report-System, auf Lucas Wunsch: "smart aber
zurechenbar" — deterministische Meilenstein-Zeitpläne statt reaktivem
Auto-Tuner)
- **Phase 2.1** Längen-balanciertes Lernen: `ReplayBuffer` trackt jetzt die
  Schlangenlänge pro Eintrag (`NStepChain`/`push()` erweitert) und erzwingt
  `balance_anteil` (Default 0.3) Mindestanteil an Zügen mit Länge ≥
  `balance_min_laenge` (30) pro Lern-Batch. `buffer_size` 100k → **1 Mio**
  (Luca-Wunsch, siehe Chat-Begründung: ~700MB RAM, 300k-Puffer rotiert bei
  15-20k Zügen/s in nur ~15-20 Sekunden komplett durch — bei 1 Mio. wächst
  das Zeitfenster auf ~1 Minute).
- **Phase 2.6** Symmetrie-Verdopplung: `ai/perception.py` hat jetzt
  `MIRROR_MAPS` (Permutation+Vorzeichen je Wahrnehmung: simple, rich,
  rich_grid5/7/9 — bewusst NICHT für full_board, siehe Code-Kommentar) +
  `mirror_perception()` + `ACTION_MIRROR`. `ReplayBuffer.sample()` spiegelt
  mit 50% Wahrscheinlichkeit (`cfg.spiegel_lernen`, Default an) Zustand +
  Folgezustand + Aktion gemeinsam. **Abnahmetest wie im Plan gefordert**:
  250 Zufalls-Spielzustände x-gespiegelt und mit `mirror_perception(...)`
  verglichen — exakte Übereinstimmung (atol 1e-6).
- **Phase 2.9** Wachsende Prüf-Notbremse: `eval_max_steps` in
  `run_evaluation()` jetzt `max(cfg.eval_max_steps, 50*(eval_best+20))` —
  deckelt zukünftigen Erfolg nicht mehr selbst.
- **Phase 1.3/2.4/2.5 — Meilenstein-Zeitpläne** (NEU gegenüber Original-Plan:
  bewusst als deterministische, an `eval_best` gekoppelte Funktionen gebaut,
  nicht als "reagiert auf Stillstand"-Auto-Tuner — das war explizit Lucas
  Entscheidung im Chat):
  - `MultiGameTrainer.formung_faktor` (Property): Näher/Weiter-Formung
    faded linear zwischen `formung_aus_ab`/`formung_null_ab` aus.
  - `MultiGameTrainer.eps_end_active` (Property): Neugier-Boden sinkt ab
    `eps_spaet_ab` auf `eps_end_spaet`.
  - `MultiGameTrainer._apply_lr_milestone()`: setzt die Optimizer-Lernrate
    auf die höchste erreichte Stufe aus `cfg.lr_meilensteine`.
  - `reward_win` (neu, Default 100): Sieg-Bonus zusätzlich zur Frucht-Belohnung.
  - Alle Schwellen sind für 17×15 (255 Zellen) kalibriert und werden über
    `_milestone_scale()` proportional zur Feldfläche skaliert — sonst auf
    kleinen Curriculum-Brettern (9×9 = 78 max. Punkte) nie erreichbar.
  - Jede Änderung landet in `trainer.milestone_log` (Episode, eval_best,
    was geändert wurde) → erscheint im Report.
- **Phase 0.1 Post-Run-Report** (`ai/dqn/report.py`, neu): `write_report()`
  erzeugt `logs/dqn-<runid>-report.json` + `-report.md` mit **Todesursachen
  nach Schlangenlängen-Eimer** (die Kernauswertung — Wand/Selbst/Verhungert/
  Sieg % + Zbuilt/Frucht je 10er-Eimer), Score-Histogramm, Todes-Positions-
  Raster für Selbstkollisionen, Q-Kalibrierung (vorhergesagter Start-Q vs.
  tatsächlicher Score), Meilenstein-Protokoll, Config-Abweichungen vom
  Standard, und automatische Diagnose-Texte (Selbstkollision >60% in einem
  Eimer ≥30, Prüfung stagniert über 5 Prüfungen, Loss steigt) — **rein
  meldend, kein Eingriff**. Aufrufstellen: `train_dqn.py` (`try/finally`,
  greift auch bei Strg+C), Dashboard bei Esc→Menü UND beim Fenster-Schließen.
  `MultiGameTrainer.write_report()` ist der Einstiegspunkt.

**Verifiziert mit Smoke-Tests** (im Chat ausgeführt, nicht Teil des Repos):
Puffer-Balance (Anteil ~50%±5 bei Ziel 50%, `_long_indices` bleibt nach
Ring-Überschreiben korrekt), Spiegel-Korrektheit (250 Vergleiche über 5
Wahrnehmungen), Spiegel-im-Sample (interne Pufferzeilen bleiben nach vielen
`sample()`-Aufrufen unverändert — keine In-Place-Mutation), alle drei
Meilenstein-Mechanismen (Property-Werte + tatsächliches Feuern in einem
echten 15.000-Tick-Lauf), Report-Erzeugung inkl. Eimer-Summen == Gesamt-
Episoden, kompletter End-to-End-Lauf mit ALLEN neuen Mechanismen
gleichzeitig + Brett-Transfer 9×9→13×11, Neuroevolution-Regressionstest
(bleibt tanh, Genom-Roundtrip unverändert), `watch_ai.py` findet/lädt den
richtigen Champion. Alle `.py`-Dateien im Projekt kompilieren fehlerfrei.

### Runde 4 (Phase 2.2 Endspiel-Curriculum, 2026-07-24 — Lucas OK nach Runde-3-Report)
Umgesetzt genau nach Spec:
- `game/snake_game.py`: neue Methode `load_snapshot(snake, fruits, direction,
  steps_since_fruit)` — strukturell wie `reset()` (der Motor legt den
  Startpunkt fest), keine neue Spielregel/Strategie, KI bekommt keine
  Zusatzinfo.
- `ai/dqn/config.py`: `curriculum_anteil: float = 0.25`.
- `ai/dqn/trainer.py`: `curriculum_path()` (eigene Datei je Brett, wie
  `champion_path()`), `_load_/_save_curriculum_snapshots()`,
  `_reset_or_curriculum()` (ersetzt `game.reset()` in `_finish_episode` UND
  bei der Initialbelegung in `__init__`). `run_evaluation()` sammelt
  waehrend jeder Pruefung pro Partie Stellungen bei Laenge 40/50/60 und
  behaelt davon nur die der BESTEN Pruefpartie (Ring, `[-200:]`,
  `models/startstellungen_{cols}x{rows}.pkl`). Pruefungen selbst bleiben
  unveraendert bei `g.reset()` (immer Laenge 3 — reiner Massstab).
- **Smoke-Tests (im Chat ausgeführt)**: `curriculum_anteil=1.0` laedt
  zuverlaessig eine injizierte Stellung (Laenge/Score korrekt); Pruefpartien
  bleiben dabei unberuehrt bei Laenge 3; mit klein gesetzten Test-Schwellen
  (4/5/6 statt 40/50/60) sammelt eine echte `run_evaluation()` tatsaechlich
  Stellungen ein, schreibt die `.pkl`-Datei, ein neuer Trainer laedt sie
  wieder — Persistenz ueber Neustarts hinweg funktioniert; Ring-Deckel bei
  200 greift. Alle geänderten Dateien kompilieren fehlerfrei.
- Effekt (noch nicht gemessen, da neu): ein Viertel der Trainingspartien
  sollte ab der ersten Pruefung mit gesammelten Stellungen direkt im
  Endspiel starten, statt die seltene "lang + wenig Platz"-Situation nur
  zufällig zu treffen — sollte laut Plan direkt gegen den in Runde 3
  gemessenen Selbstkollisions-Anstieg (58%→81% mit der Länge) wirken.

**Bewusst NICHT umgesetzt** (auf der Sperrliste C-9 oder Zeitgründe):
- **Auto-Tuner** (reaktiv auf Stillstand) — von Luca explizit abgelehnt.
- **Phase 0.2** Einsperr-Analyse (Flood-Fill NUR als Post-Mortem-Telemetrie).
- **Phase 0.4/0.5** Meilenstein-Bibliothek (Checkpoints pro Füllgrad), A/B-Runner-Skript.
- **Phase 0.3 (Rest)**: doppelte Bestätigungs-Prüfung vor Champion-Speicherung.
- **Phase 1.2** Dueling-Kopf.
- **Phase 2.7/2.8**: gamma-A/B, PER-Retest (reine Experimente, kein Code nötig).
- **Phase 3**: alle Performance-Optimierungen (Board-Zeichnen-Drossel etc.).
- **Phase 4**: CNN (separat geplant, erst nach Messung von 0-2).

**Nächster empfohlener Schritt:** frischen (oder `--weiter`-)Lauf auf 17×15
mit den Runde-2-Standardwerten starten (`rich_grid7`, `fruit_count=3`,
`eps_decay_steps=80000` — Runde 3 war ein Abweichler und daher kein sauberer
Vergleich) und laufen lassen, bis die Prüfung sich wieder setzt. Danach
Report lesen und gezielt vergleichen: sinkt die Selbstkollisionsrate in den
hohen Längen-Eimern (60+) gegenüber Runde 3? Falls ja, wirkt das Curriculum
wie erhofft — weiter laufen lassen. Falls kein Unterschied sichtbar wird,
ist Phase 4 (CNN) der nächste Kandidat, da dann eher die Wahrnehmung selbst
(nicht die Trainingsverteilung) die Grenze ist.

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
