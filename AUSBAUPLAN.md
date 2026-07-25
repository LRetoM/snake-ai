# AUSBAUPLAN — Die letzten 20-30% Richtung 100% Feldfüllung

Stand: 2026-07-25. Umsetzungs-Spezifikation für die fünf Netz-/Trainings-
Upgrades aus dem Chat (mitwachsendes Curriculum, Dueling, Noisy Nets, CNN,
Distributional RL). **Noch NICHTS davon ist umgesetzt** — dieses Dokument
ist der exakte Bauplan, dem später Phase für Phase gefolgt wird
("setze Phase B um" reicht dann als Auftrag).

Basis-Messung beim Schreiben dieses Plans: Auto-Tuner-Bestmarke 146-152
(≈58-61% Füllung im Schnitt, Prüfung ohne Zufall) mit
`rich_grid9, spiegel_lernen=False, fruit_count=10, target_update=500,
curriculum_anteil=1.0, n_step=5, pfad_fokus=0.7, gamma=0.99, batch_size=512`.
Beste faire Einzelpartie: 207 (82%).

---

## Regeln für ALLE Phasen (gelten immer, nicht verhandelbar)

1. **Leitplanke bleibt**: kein Pathfinding, kein Flood-Fill, keine Heuristik
   als KI-Eingabe oder im Handeln. Alle Upgrades hier ändern nur, WIE das
   Netz lernt/sieht — nie, WAS ihm vorgesagt wird.
2. **Ein Unterschied pro Vergleich** (TRAININGSPLAN.md C-10): jede Phase
   wird einzeln per A/B gemessen (gleiches Wanduhr-Budget, fester Seed,
   `--override`), bevor sie Default wird oder in den Tuner-Suchraum kommt.
   Sieger nur bei Bestätigung über ≥2 Seeds.
3. **Checkpoint-Kompatibilität**: JEDES neue Config-/Netz-Feld wird
   (a) im Checkpoint-Payload gespeichert (`agent.save_checkpoint`),
   (b) beim Laden mit Rückwärts-Fallback gelesen (fehlt das Feld in einem
   alten Checkpoint → alter Standardwert, z.B. `dueling=False`),
   (c) bei Mismatch mit klarer Fehlermeldung abgelehnt (wie heute schon
   `activation`/`hidden` in `MultiGameTrainer._resume`).
   Betroffene Ladepfade IMMER alle drei: `trainer._resume`,
   `load_champion_config`, `watch_ai.py`.
4. **Neuroevolution bleibt unangetastet**: `SnakeNet` wird nur ERWEITERT
   (neue Parameter mit Defaults = heutiges Verhalten). Der Genom-Roundtrip
   (`get_genome`/`load_genome`) muss bei Default-Argumenten byte-identisch
   bleiben — Regressionstest in jeder Phase, die `torch_bridge.py` anfasst.
5. Nach jedem Schritt: `python -m py_compile` auf alle geänderten Dateien
   + die je Phase definierten Abnahmetests. Kommentarstil des Projekts
   (deutsch, erklärend, Leitplanken-Hinweise). Nichts committen ohne OK.
6. **Tuner-Integration zuletzt**: neuer Schalter kommt erst in
   `auto_tuner.SUCHRAUM`, nachdem der Einzel-A/B ihn als mindestens
   gleichwertig gezeigt hat (sonst verbrennt der Tuner Stunden an einem
   kaputten Feature).

**Empfohlene Reihenfolge**: A → B → C → D → E (aufsteigender Aufwand;
A/B/C sind unabhängig voneinander, D und E bauen auf B auf, weil Dueling-
und Distributional-Kopf sich den Netz-Umbau teilen).

---

## Phase A — Mitwachsendes Endspiel-Curriculum (kleinster Eingriff)

**Problem**: `_CURRICULUM_LENGTHS = (40, 50, 60)` in `ai/dqn/trainer.py` ist
fest. Der Bot erreicht im Schnitt schon ~150 — Stellungen bei Länge 40-60
sind für ihn längst „Mittelspiel", geübt wird nicht mehr an seiner echten
Grenze.

**Umsetzung**:
- `ai/dqn/config.py`, neue Felder (mit Erklär-Kommentar):
  - `curriculum_schwellen_relativ: tuple = (0.5, 0.65, 0.8)` — Anteile vom
    Lauf-Bestwert.
  - `curriculum_schwelle_min: int = 40` — Untergrenze, solange der Bot noch
    schwach ist (entspricht heutigem Verhalten am Anfang).
- `ai/dqn/trainer.py`:
  - Modul-Konstante `_CURRICULUM_LENGTHS` ersetzen durch Methode
    `MultiGameTrainer._curriculum_schwellen(self) -> tuple[int, int, int]`:
    `max(cfg.curriculum_schwelle_min, int(self.eval_best_run * f))` je
    `f in cfg.curriculum_schwellen_relativ`, zusätzlich gedeckelt auf
    `cols*rows - 10` (kurz vor Sieg macht ein Snapshot keinen Sinn).
  - `run_evaluation()`: benutzt diese Methode statt der Konstante
    (Stelle: `next_target`/`curriculum_hits`-Schleife).
  - Snapshots bekommen zusätzlich `"laenge": game.length` ins Dict
    (rein informativ für Report/Debug; `load_snapshot` ignoriert es).
- `ai/dqn/report.py`: im Curriculum-Block zusätzlich die Verteilung der
  Snapshot-Längen ausgeben (min/median/max der `len(snake)` im Vorrat) —
  zeigt sofort, ob das Mitwachsen wirkt.

**Abnahme**:
1. `eval_best_run=0` → Schwellen (40, 40, 40); `=150` → (75, 97, 120);
   `=300` auf 17×15 → Deckel greift (245).
2. Bestehender Curriculum-Smoke-Test (Anteil 1.0 startet lang, Prüfung
   startet kurz) bleibt grün.
3. Trap-Erkennung unverändert funktionsfähig (bestehende Tests).

**Messung**: A/B mit Tuner-Bestconfig ± mitwachsend, 2 Seeds, 20 Min.

---

## Phase B — Dueling-Kopf

**Idee**: Das Netz trennt „wie gut ist diese Lage überhaupt" (Value, 1 Wert)
von „wie viel besser ist Aktion X als der Durchschnitt" (Advantage, 3
Werte): `Q = V + A − mean(A)`. Hilft besonders dort, wo fast jede Aktion
gleich gut ist (offenes Feld) vs. genau eine überlebt (Engpass).

**Umsetzung**:
- `ai/torch_bridge.py`, `SnakeNet.__init__` erweitern:
  `dueling: bool = False` (Default = heutiges Verhalten, Neuroevolution
  ruft weiter ohne Argument auf).
  - Wenn False: alles exakt wie heute (`self.out`).
  - Wenn True: statt `self.out` zwei Köpfe auf der letzten Hidden-Größe:
    `self.value_head = nn.Linear(prev, 1)`,
    `self.adv_head = nn.Linear(prev, OUTPUT_SIZE)`;
    `forward`: `q = v + a - a.mean(dim=1, keepdim=True)`.
  - `get_genome`/`load_genome`: bei `dueling=True` mit klarem
    `RuntimeError` ablehnen („Dueling-Netze haben kein Neuroevolution-
    Genom-Layout") — verhindert stilles falsches Mapping.
- `ai/dqn/config.py`: `dueling: bool = False` (Default aus, bis A/B ihn
  belegt — bewusst ANDERS als die alte Spec in TRAININGSPLAN.md 1.2, wir
  messen erst seit Runde 5 sauber).
- `ai/dqn/agent.py`: beide Netz-Konstruktionen bekommen
  `dueling=getattr(cfg, "dueling", False)`; `save_checkpoint`-Payload
  speichert `"dueling"`.
- `ai/dqn/trainer.py` `_resume`: Mismatch-Prüfung analog `activation`
  (Checkpoint-Fallback: `checkpoint.get("dueling", False)`).
- `watch_ai.py`: beim Netz-Bauen `dueling=checkpoint.get("dueling", False)`
  durchreichen.

**Abnahme**:
1. Forward-Shape (batch, 3) für beide Modi; bei dueling=True gilt
   `Q.mean(dim=1) ≈ V` (Advantage-Mittel ist 0).
2. Checkpoint-Roundtrip dueling=True; ALTER Checkpoint (ohne Feld) lädt
   als dueling=False und spielt unverändert.
3. Genom-Roundtrip-Regressionstest bei Default-Argumenten (Regel 4).
4. Trainings-Smoke 500 Ticks ohne Loss-Explosion.

**Messung**: A/B dueling False/True auf Tuner-Bestconfig, 2 Seeds, 20 Min.
Danach ggf. `"dueling": [False, True]` in `auto_tuner.SUCHRAUM`.

---

## Phase C — Noisy Nets (Exploration im Netz statt Würfel-Zug)

**Problem heute**: Epsilon-greedy würfelt pro Zug. Ein einzelner
Zufallszug tötet regelmäßig eine 300-Züge-Partie — je besser der Bot,
desto teurer jeder Würfelwurf (genau deshalb existiert schon
`eps_end_spaet`). Noisy Nets ersetzen den Würfel durch lernbares Rauschen
IN den Netz-Gewichten: das Netz exploriert dort viel, wo es unsicher ist,
und wird von selbst deterministischer, wo es sicher ist.

**Umsetzung**:
- `ai/torch_bridge.py`, neue Klasse `NoisyLinear(nn.Module)`
  (Factorised-Gaussian-Standard, sigma_init=0.5):
  - Parameter: `weight_mu/weight_sigma/bias_mu/bias_sigma`; Buffer
    `weight_eps/bias_eps`; Methode `resample_noise()`; `forward`: im
    train()-Modus mu+sigma*eps, im eval()-Modus NUR mu (deterministisch —
    dadurch bleiben Prüfungen automatisch der ehrliche, rauschfreie Wert).
- `SnakeNet.__init__`: `noisy: bool = False`. Wenn True: alle
  `nn.Linear`-Schichten (hidden + Kopf bzw. beide Dueling-Köpfe) werden
  `NoisyLinear`. Methode `resample_noise()` am Netz (ruft alle Schichten).
  Genom-Pfad wie bei Dueling: `RuntimeError` bei noisy=True.
- `ai/dqn/config.py`: `noisy: bool = False`.
- `ai/dqn/agent.py`:
  - Netz-Bau mit `noisy=...`; Checkpoint-Feld `"noisy"`.
  - `act_batch`: wenn noisy → VOR dem Forward `policy_net.resample_noise()`
    und epsilon KOMPLETT ignorieren (kein Würfeln); sonst wie heute.
  - `learn()`: wenn noisy → vor jedem der beiden Forwards (policy und
    target) `resample_noise()` auf dem jeweiligen Netz (target braucht
    eigenes Rauschen — Standard bei Rainbow).
- `ai/dqn/trainer.py`:
  - wenn `cfg.noisy`: `self.epsilon` fest auf 0.0 setzen (Anzeige und
    Formel), die eps-Meilensteine (eps_end_active) sind dann wirkungslos —
    mit Kommentar dokumentieren, NICHT löschen.
  - `run_evaluation`: nichts nötig — dort läuft `act_batch(states, 0.0)`
    und die Netze stehen in eval()… ACHTUNG: `policy_net` steht im
    Training nie auf eval(). Deshalb explizit: in `run_evaluation` um den
    Prüf-Block `policy_net.eval()` … `policy_net.train()` setzen (heute
    egal, mit Noisy PFLICHT, sonst prüft man mit Rauschen).
  - `_resume`: Mismatch-Prüfung `noisy`.
- `watch_ai.py`: Netz nach dem Laden auf `.eval()` (rauschfrei zuschauen);
  `noisy=checkpoint.get("noisy", False)`.

**Abnahme**:
1. noisy=True: zwei Forwards mit `resample_noise()` dazwischen liefern
   VERSCHIEDENE Q-Werte (train-Modus); im eval()-Modus identische.
2. Trainings-Smoke 1000 Ticks: Loss endlich, Score steigt über Zufall.
3. Prüfung (run_evaluation) mit noisy=True ist reproduzierbar
   deterministisch bei festem Spiel-Seed.
4. Alter Checkpoint lädt (noisy fehlt → False), Genom-Regressionstest.

**Messung**: A/B noisy False/True. Wichtig: noisy=True ersetzt die
komplette Epsilon-Mechanik — im A/B-Arm noisy also `eps_start=0`
mitsetzen, damit nicht beides gleichzeitig wirkt.

---

## Phase D — CNN-Wahrnehmung („das ganze Brett als Bild") — der große Sprung

**Problem**: `rich_grid9` sieht 9×9 um den Kopf. Fallen, die sich außerhalb
schließen, sind unsichtbar — die Kernursache der verbleibenden
Selbstkollisionen. Ein CNN sieht das GANZE Brett räumlich.

**Design-Entscheidungen (bewusst so und nicht anders)**:
- Die Wahrnehmung bleibt ein FLACHER Vektor (damit ReplayBuffer,
  n-Schritt-Ketten und Spiegel-Mechanik UNVERÄNDERT weiterlaufen); das
  NETZ formt ihn intern zu Kanälen um. Kein Umbau an memory.py nötig.
- Kanäle sind ROHE Sicht, keine Heuristik: (0) eigener Körper, (1) Kopf,
  (2) Frucht — je Zelle 0/1. Dazu 6 Skalare: Blickrichtung one-hot (4),
  Länge normiert, Hunger. KEIN Flood-Fill-Kanal, KEIN „sicherer Weg"-Kanal.
- Layout des Vektors (fest dokumentieren!):
  `[skalare(6)] + [kanal0 zeilenweise (y, dann x)] + [kanal1] + [kanal2]`
  → `size = 6 + 3*cols*rows` (17×15: 771).
- Anders als `full_board` bekommt diese Wahrnehmung eine SPIEGELUNG:
  Spalten-Flip `x → cols-1-x` ist eine reine Permutation je Kanal, der
  Kopf ist als Kanal kodiert (keine normierte Koordinate im Weg), in den
  Skalaren tauschen nur Richtung links↔rechts. Damit funktioniert
  `spiegel_lernen` auch hier.

**Umsetzung**:
- `ai/perception.py`:
  - `make_cnn_perception(cols, rows) -> (fn, size, labels)`; Registrierung
    in `BOARD_DEPENDENT_PERCEPTIONS["cnn_board"]`.
  - Spiegel-Eintrag: da brettabhängig, braucht `MIRROR_MAPS` eine
    Erweiterung: `mirror_perception(name, vector)` fällt heute auf None
    zurück — neu: Funktion `make_cnn_mirror(cols, rows)` und im
    ReplayBuffer-Bau (`make_buffer`) für brettabhängige Wahrnehmungen die
    Map zur Laufzeit erzeugen (Signatur-Erweiterung:
    `make_buffer(cfg, state_size, rng, mirror_entry=None)`; `trainer.py`
    reicht den passenden Eintrag durch).
- `ai/torch_bridge.py`, neue Klasse `SnakeConvNet(nn.Module)`:
  - `__init__(cols, rows, skalare=6, kanaele=3, activation="relu",
    dueling=False, noisy=False)`.
  - Architektur: `Conv2d(3→32, 3, padding=1)` → act →
    `Conv2d(32→64, 3, padding=1)` → act →
    `AdaptiveAvgPool2d((11, 11))` → Flatten (64*121) →
    concat Skalare → `Linear(7750 → 256)` → act → Kopf
    (normal 3 / Dueling V+A; Linear-Typ je `noisy`).
  - Das adaptive Pooling macht das Netz BRETTGRÖSSEN-TOLERANT →
    Klein-Feld-Curriculum (9×9 → 17×15) funktioniert mit Gewichts-
    Transfer. Kompromiss (dokumentieren): kostet etwas Ortsauflösung;
    ein späteres A/B darf eine Variante ohne Pooling (fixes Brett) testen.
  - `forward(x)`: `skalare = x[:, :6]`, `brett = x[:, 6:].view(-1, 3,
    rows, cols)` (rows/cols aus dem Konstruktor).
  - Kein Genom-Pfad (RuntimeError wie oben).
- `ai/dqn/config.py`: `network: str = "mlp"` (`"cnn"` aktiviert
  SnakeConvNet). Validierung: `network="cnn"` erfordert
  `perception="cnn_board"` (und umgekehrt) — klare Fehlermeldung beim
  Trainer-Bau, nicht erst beim Forward.
- `ai/dqn/agent.py`: Netz-Fabrik-Funktion `_baue_netz(cfg, input_size)`
  (wählt SnakeNet/SnakeConvNet); Checkpoint speichert `"network"`,
  `"grid_cols"/"grid_rows"` sind schon drin.
- `ai/dqn/trainer.py`: `_resume`-Mismatch für `network`; beim
  Brett-Transfer mit CNN ist `input_size` unterschiedlich, die
  CONV-Gewichte aber übertragbar → Transfer-Laden mit
  `strict=False`-Strategie NUR für den dokumentierten Fall
  „gleiches Netz, andere Brettgröße" (erste FC-Schicht nach dem Pooling
  hat dank AdaptiveAvgPool dieselbe Größe — es bleibt also sogar strict
  ladbar; genau das im Abnahmetest beweisen).
- `watch_ai.py`: `network`-Feld lesen, Netz passend bauen.
- Performance-Erwartung dokumentieren: CNN ist auf CPU je Zug deutlich
  teurer. Erster Messlauf mit `torch_threads=0` (auto), `num_games` ggf.
  runter (8) — der Tuner darf das später ausbalancieren.

**Abnahme**:
1. Wahrnehmungs-Test: für 200 Zufallszustände auf 9×9/13×11/17×15:
   `fn(game).shape == (size,)`, Kanal-Summen stimmen (Körper-Kanal ==
   Schlangenlänge, Kopf-Kanal == 1, Frucht-Kanal == fruit_count).
2. Spiegel-Test wie in TRAININGSPLAN 2.6: `mirror(perceive(game)) ==
   perceive(mirrored_game)` (atol 1e-6) inkl. Aktions-Tausch.
3. Forward-Shapes auf allen drei Brettern mit DENSELBEN Gewichten
   (Brettgrößen-Toleranz), Checkpoint-Roundtrip, 9×9-Checkpoint lädt
   strict auf 17×15-Netz.
4. Trainings-Smoke 2000 Ticks auf 9×9: Score steigt über Zufallsniveau;
   Züge/s im Report notieren (Erwartungswert setzen für spätere
   Regressionen).

**Messung**: erst 9×9-Kurzvergleich CNN vs. rich_grid9 (schneller
Signal-Check), dann 17×15 A/B 2 Seeds × 30-45 Min (CNN braucht mehr Zeit
pro Erkenntnis — kurze Fenster würden es systematisch benachteiligen,
siehe Sprinter-Problem in TRAININGSPLAN Runde 5d).

---

## Phase E — Distributional RL (QR-DQN)

**Warum QR-DQN und nicht C51**: C51 braucht feste Wert-Grenzen
[Vmin, Vmax] — unsere Returns wachsen mit jedem Fortschritt (Q-Mittel ist
im Tuner-Lauf schon von 20 auf 150+ gestiegen), die Grenzen müssten
laufend nachkalibriert werden. Quantile Regression (QR-DQN) hat keine
Grenzen und ist genau dafür robuster. Gleiche Kern-Idee: das Netz lernt
die VERTEILUNG der möglichen Zukunft statt nur ihren Mittelwert — wichtig
bei uns, wo derselbe Zug mal +50 und mal Tod bedeutet.

**Umsetzung**:
- `ai/dqn/config.py`: `distributional: bool = False`,
  `quantile_anzahl: int = 32`.
- `ai/torch_bridge.py`: `SnakeNet`/`SnakeConvNet` bekommen
  `quantile: int = 0` (0 = normale Q-Werte). Wenn >0: Kopf gibt
  `(batch, 3*quantile)` aus, `forward` reshaped zu `(batch, 3, quantile)`.
  Dueling-Kombination: Value-Kopf → `(batch, 1, quantile)`, Advantage →
  `(batch, 3, quantile)`, `Q = V + A - A.mean(dim=1, keepdim=True)`.
  - NEU, zentral: Methode `q_values(x)` an BEIDEN Netzklassen — liefert
    IMMER `(batch, 3)` (bei quantile>0 das Mittel über die Quantile).
    Alle „wie gut ist Aktion X"-Konsumenten (`act_batch`,
    Q-Kalibrierung in `run_evaluation`, `watch_ai`) stellen auf
    `q_values()` um — dadurch bleibt ihr Code für alle Netz-Varianten
    identisch.
- `ai/dqn/agent.py`, `learn()` verzweigt bei `cfg.distributional`:
  - Ziel: `theta_target = r + disc * theta_target_next * (1-done)` je
    Quantil; Double-DQN: beste Folge-Aktion per
    `policy_net.q_values(s2).argmax`, deren Quantile aus dem Target-Netz.
  - Loss: Quantile-Huber (kappa=1.0) über alle Quantil-Paare
    (Standard-Formel, τ_i = (2i+1)/2N).
  - PER-Priorität: `|mean TD|` über die Quantile (kompatibel zum
    bestehenden `update_priorities`).
  - `last_mean_q` aus `q_values` — Dashboard/Reports laufen unverändert.
- Checkpoint-Felder `"distributional"`, `"quantile_anzahl"` + Mismatch-
  Prüfung in `_resume`; `watch_ai` nutzt ohnehin nur `q_values`.

**Abnahme**:
1. Shapes: forward `(B,3,32)`, `q_values` `(B,3)`; Dueling-Kombi ok.
2. Synthetischer Lern-Test: konstruierter Puffer mit bekannter bimodaler
   Belohnung → gelernte Quantile decken beide Modi ab (grober Check:
   min/max-Quantil klar getrennt).
3. Trainings-Smoke 1000 Ticks; Q-Kalibrierung im Report bleibt plausibel.
4. Alter Checkpoint lädt (Felder fehlen → aus/0).

**Messung**: A/B distributional False/True auf der dann aktuellen
Bestconfig, 2 Seeds, ≥30 Min (auch hier: Verteilungs-Lernen braucht Zeit).

---

## Abhängigkeits-/Reihenfolge-Übersicht

| Phase | Aufwand | hängt ab von | Haupt-Dateien |
|---|---|---|---|
| A Curriculum wächst mit | klein | — | config, trainer, report |
| B Dueling | klein-mittel | — | torch_bridge, config, agent, trainer, watch_ai |
| C Noisy Nets | mittel | — (kombinierbar mit B) | torch_bridge, config, agent, trainer, watch_ai |
| D CNN | groß | B empfohlen zuerst | perception, torch_bridge, memory(make_buffer-Signatur), config, agent, trainer, watch_ai |
| E QR-DQN | groß | B (geteilter Kopf-Umbau) | torch_bridge, config, agent |

Nach jeder als Sieger bestätigten Phase: neuen Tuner-Lauf
(`--basis-von` letzte Bestconfig) mit dem neuen Schalter im Suchraum —
die Hyperparameter-Balance kann sich mit jedem Architektur-Upgrade
verschieben (z.B. braucht CNN evtl. andere lr/batch_size).

## Bewusst NICHT im Plan (Sperrliste)

- Flood-Fill/Erreichbarkeits-Kanal als Netz-Eingabe (Heuristik-Leck).
- Sofort-Tod-Maskierung (C-9, braucht Lucas explizites OK).
- Hamiltonkreis-/Pathfinding-Fallback jeder Art.
- Automatisches Hyperparameter-Tuning WÄHREND eines Laufs (der Tuner
  arbeitet bewusst ZWISCHEN Läufen).
