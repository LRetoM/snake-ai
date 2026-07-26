"""Der DQN-Agent -- das Gehirn: entscheiden (act) und lernen (learn).

Das Prinzip in einem Absatz
---------------------------
Das Netz ist ein "Notizbuch mit Bewertungen". Eingabe: die 11 Wahrnehmungszahlen
(ai/perception.py). Ausgabe: drei Zahlen, die Q-Werte -- einer pro Aktion
(geradeaus / links / rechts). Ein Q-Wert ist eine SCHAETZUNG der Frage:

    "Wenn ich in dieser Situation diese Aktion waehle -- wie viel Belohnung
     bringt mir das insgesamt noch ein, bis die Partie vorbei ist?"

Gespielt wird dann einfach: nimm die Aktion mit dem hoechsten Q-Wert.

Woher kommen die richtigen Q-Werte? (Bellman-Gleichung)
-------------------------------------------------------
Am Anfang sind alle Schaetzungen zufaelliger Unsinn. Sie werden mit dieser einen
Regel schrittweise wahr:

    Q(Situation, Aktion)  =  sofortige Belohnung
                             + gamma * bester Q-Wert der FOLGE-Situation

Auf Deutsch: "Was dieser Zug wert ist = was er sofort bringt, plus was ich in
der Situation danach bestenfalls noch holen kann (leicht abgewertet)."

Das erzeugt einen Domino-Effekt rueckwaerts durch die Zeit:
Der Zug IN die Wand bekommt sofort -10, seine Schaetzung wird also klar negativ.
Beim naechsten Lernen merkt der Zug DAVOR: "die Situation danach ist -10 wert" --
also wird auch er negativ bewertet. Und der davor. So wandert die Erkenntnis
"in Richtung Wand fahren ist eine schlechte Idee" Schritt fuer Schritt nach
hinten, obwohl wir der KI nie gesagt haben, was eine Wand ist. Genau deshalb
wird sie mit jeder Erfahrung besser: die Schaetzungen naehern sich der Wahrheit.

gamma (hier 0.9) ist die "Ungeduld": Belohnung in 10 Zuegen zaehlt nur noch
0.9^10 ~ 0.35 so viel wie sofortige. Ohne dieses Abwerten wuerde die Rechnung
bei langen Partien ins Unendliche laufen.

Die drei Stabilisatoren
-----------------------
(a) **Experience Replay** -- das Tagebuch, siehe memory.py.
(b) **Target-Netz** -- eine EINGEFRORENE Kopie des Netzes, die das Lernziel
    liefert. Ohne sie wuerde das Netz sein eigenes Ziel jagen, waehrend es sich
    bewegt: als wuerde man eine Zielscheibe treffen wollen, die man bei jedem
    Schuss selbst verschiebt. Die Kopie wird nur alle paar tausend Lernschritte
    aktualisiert -- feststehende Zielscheibe, ruhiges Lernen.
(c) **Epsilon-greedy** -- Neugier. Mit Wahrscheinlichkeit epsilon wuerfelt die
    KI einen Zufallszug statt des besten bekannten. Am Anfang epsilon=1.0
    (alles ausprobieren), spaeter 0.02 (dem Gelernten vertrauen). Ohne Neugier
    wuerde die KI die erste halbwegs brauchbare Masche fuer immer wiederholen
    und nie merken, dass es viel besser geht.

Alle Spiele teilen sich EIN Exemplar dieses Agenten -- deshalb entscheidet
`act_batch()` fuer alle Schlangen gleichzeitig in EINEM Netz-Durchlauf. Das ist
nicht nur bequem, sondern auch deutlich schneller als 5 Einzelaufrufe.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.network import OUTPUT_SIZE
from ai.torch_bridge import SnakeConvNet, SnakeNet


def _baue_netz(cfg, input_size: int):
    """Netz-Fabrik fuer alle Architektur-Schalter (AUSBAUPLAN.md).

    Waehlt MLP (SnakeNet) oder CNN (SnakeConvNet) und reicht die
    Kopf-Schalter (dueling/noisy/quantile) durch. Alle Schalter Default AUS
    ergeben exakt das bisherige Netz -- alte Checkpoints laden unveraendert.
    """
    quantile = (getattr(cfg, "quantile_anzahl", 32)
                if getattr(cfg, "distributional", False) else 0)
    dueling = getattr(cfg, "dueling", False)
    noisy = getattr(cfg, "noisy", False)
    activation = getattr(cfg, "activation", "tanh")
    if getattr(cfg, "network", "mlp") == "cnn":
        return SnakeConvNet(cfg.grid_cols, cfg.grid_rows,
                            activation=activation, dueling=dueling,
                            noisy=noisy, quantile=quantile,
                            kanaele=tuple(getattr(cfg, "cnn_kanaele", (16, 32))),
                            pool=getattr(cfg, "cnn_pool", 6))
    return SnakeNet(cfg.hidden, input_size, activation,
                    dueling=dueling, noisy=noisy, quantile=quantile)


def pick_device(cfg=None) -> torch.device:
    """CUDA-Grafikkarte, falls vorhanden -- sonst MPS beim CNN, sonst CPU.

    Die Groesse des Netzes entscheidet, ob sich die Grafikkarte lohnt, und die
    beiden Faelle liegen bei uns weit auseinander (gemessen 2026-07-26 auf M4,
    Batch 256, Nachtlauf-Konfiguration -- siehe TRAININGSPLAN Runde 9):

      MLP (rich_grid9, 120 Eingaenge):  CPU 1.05 ms  |  MPS 3.31 ms  -> CPU
      CNN (cnn_board, ganzes Brett):    CPU 40.7 ms  |  MPS 16.0 ms  -> MPS

    Beim MLP frisst der Weg zur Grafikkarte und zurueck mehr, als die Rechnung
    selbst kostet -- deshalb bleibt es dort (wie bisher) bei der CPU. Beim CNN
    ist es umgekehrt: echte Bild-Faltungen sind genau die Rechenart, fuer die
    Grafikkarten gebaut sind, und der Transfer faellt daneben kaum ins Gewicht
    (2.5x schneller inklusive Transfer, komplett gemessen).

    Ohne `cfg` (z.B. aus watch_ai) bleibt es bei der CPU: das Zuschauen
    rechnet einzelne Zuege, da lohnt die Grafikkarte nie.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (cfg is not None and getattr(cfg, "network", "mlp") == "cnn"
            and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


def tune_threads(input_size: int, hidden, batch_size: int) -> int:
    """Misst kurz, mit wie vielen CPU-Threads ein Lernschritt am schnellsten ist.

    Warum ueberhaupt? Ein Lernschritt ist eine Matrizenrechnung. PyTorch kann die
    auf mehrere Kerne verteilen -- das lohnt sich aber nur, wenn die Rechnung
    gross genug ist. Bei unseren kleinen Netzen verbringen viele Threads mehr
    Zeit mit Absprache als mit Rechnen. Welche Zahl gewinnt, haengt vom Rechner
    ab (Mac vs. Windows-Laptop koennen sich um das Fuenffache unterscheiden),
    also probieren wir es einmal in unter einer Sekunde aus, statt zu raten.
    """
    if torch.cuda.is_available():
        return torch.get_num_threads()   # auf der Grafikkarte irrelevant

    import os as _os
    max_threads = min(8, _os.cpu_count() or 1)
    candidates = sorted({1, 2, 4, max_threads})
    x = torch.randn(batch_size, input_size)
    target = torch.randn(batch_size, OUTPUT_SIZE)

    best_threads, best_time = 1, float("inf")
    for n in candidates:
        torch.set_num_threads(n)
        net = SnakeNet(hidden, input_size)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(3):                      # warmlaufen
            opt.zero_grad(); F.mse_loss(net(x), target).backward(); opt.step()
        t0 = time.perf_counter()
        for _ in range(12):
            opt.zero_grad(); F.mse_loss(net(x), target).backward(); opt.step()
        dt = time.perf_counter() - t0
        if dt < best_time:
            best_threads, best_time = n, dt

    torch.set_num_threads(best_threads)
    return best_threads


class DQNAgent:
    """Haelt beide Netze (policy + target), den Optimierer und die Lernlogik."""

    def __init__(self, cfg, input_size: int, seed: int | None = None) -> None:
        self.cfg = cfg
        self.input_size = input_size
        self.device = pick_device(cfg)
        self.rng = np.random.default_rng(seed)

        # Wie viele CPU-Threads soll PyTorch benutzen? Das ist der mit Abstand
        # groesste Geschwindigkeits-Hebel und je nach Rechner voellig
        # unterschiedlich: mal ist 1 Thread am schnellsten (kleine Netze, die
        # Abstimmung zwischen Threads kostet mehr als sie bringt), mal 4.
        # Deshalb wird es bei torch_threads=0 einmalig GEMESSEN statt geraten.
        # Auf einer Grafikkarte (CUDA/MPS) entfaellt das: dort rechnen keine
        # CPU-Threads, die Messung wuerde nur Startzeit kosten und die
        # gefundene Zahl nichts bewirken.
        if self.device.type != "cpu":
            pass
        elif cfg.torch_threads:
            torch.set_num_threads(int(cfg.torch_threads))
        else:
            self.threads = tune_threads(input_size, cfg.hidden, cfg.batch_size)
        self.threads = torch.get_num_threads()

        # policy_net = das Netz, das entscheidet UND trainiert wird.
        # Gebaut ueber die Fabrik oben -- die kennt alle Architektur-Schalter
        # (mlp/cnn, dueling, noisy, distributional).
        self.policy_net = _baue_netz(cfg, input_size).to(self.device)
        # target_net = die eingefrorene Kopie, die nur das Lernziel liefert.
        self.target_net = _baue_netz(cfg, input_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # WICHTIG bei Noisy Nets: das Target-Netz darf NICHT in eval() stehen
        # (dann waere sein Lernziel rauschfrei-deterministisch, waehrend das
        # Policy-Netz mit Rauschen lernt -- Rainbow nutzt verrauschte Ziele).
        # Ohne Noisy ist train/eval bei unseren Schichten identisch, das
        # bisherige eval() war reine Konvention -- train() aendert dort nichts.
        self.target_net.train()
        self.noisy = getattr(cfg, "noisy", False)
        self.distributional = getattr(cfg, "distributional", False)
        self.quantile = getattr(cfg, "quantile_anzahl", 32) if self.distributional else 0

        # Adam = ein bewaehrter Optimierer ("wie stark drehe ich an welchem
        # Regler"). Er passt die Schrittweite pro Gewicht automatisch an.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=cfg.learning_rate)

        self.learn_steps = 0          # Anzahl durchgefuehrter Lernschritte
        self.last_mean_q = 0.0        # Diagnose fuers Dashboard

    # ------------------------------------------------------------------ #
    # Entscheiden
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act_batch(self, states: np.ndarray, epsilon: float) -> np.ndarray:
        """Waehlt fuer ALLE Spiele gleichzeitig eine Aktion (epsilon-greedy).

        states: Array der Form (anzahl_spiele, Wahrnehmungsgroesse).
        Rueckgabe: Array mit einem Aktionsindex (0/1/2) pro Spiel.

        `torch.no_grad()` heisst: "nur ausrechnen, nichts zum spaeteren Lernen
        mitschreiben" -- beim reinen Entscheiden spart das viel Rechenzeit.
        """
        # Noisy Nets (AUSBAUPLAN C): frisches Rauschen je Entscheidung -- die
        # Erkundung passiert dann IM Netz, nicht per Wuerfel weiter unten.
        if self.noisy:
            self.policy_net.resample_noise()

        tensor = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
        # q_values() statt forward(): liefert fuer ALLE Netz-Varianten
        # (klassisch/dueling/distributional) einheitlich (batch, 3).
        q_values = self.policy_net.q_values(tensor.to(self.device))

        # Bester bekannter Zug pro Spiel.
        actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)
        self.last_mean_q = float(q_values.max(dim=1).values.mean().item())

        # ... und jetzt die Neugier: bei manchen Spielen stattdessen wuerfeln.
        # Bei Noisy Nets ENTFAELLT das komplett -- das Rauschen im Netz IST
        # die Erkundung; ein zusaetzlicher Wuerfel wuerde doppelt stoeren.
        if not self.noisy:
            explore = self.rng.random(actions.shape[0]) < epsilon
            n_explore = int(explore.sum())
            if n_explore:
                actions[explore] = self.rng.integers(0, OUTPUT_SIZE, size=n_explore)
        return actions

    # ------------------------------------------------------------------ #
    # Lernen
    # ------------------------------------------------------------------ #
    def learn(self, buffer) -> float | None:
        """Ein (oder mehrere) Lernschritte aus dem Tagebuch. Gibt den Loss zurueck.

        Rueckgabe None, solange noch zu wenig Erfahrung gesammelt wurde --
        aus 50 Erinnerungen zu lernen waere wie eine Meinung aus zwei Sekunden
        Zuschauen zu bilden.
        """
        cfg = self.cfg
        if len(buffer) < cfg.min_buffer:
            return None

        last_loss = None
        for _ in range(cfg.train_iters_per_step):
            (states, actions, rewards, next_states, dones,
             discounts, indices, weights) = buffer.sample(cfg.batch_size)

            s = torch.from_numpy(states).to(self.device)
            a = torch.from_numpy(actions).to(self.device)
            r = torch.from_numpy(rewards).to(self.device)
            s2 = torch.from_numpy(next_states).to(self.device)
            d = torch.from_numpy(dones).to(self.device)
            disc = torch.from_numpy(discounts).to(self.device)
            w = torch.from_numpy(weights).to(self.device)

            # Noisy Nets: frisches Rauschen fuer BEIDE Netze je Lernschritt
            # (das Target braucht sein EIGENES Rauschen -- Rainbow-Standard).
            if self.noisy:
                self.policy_net.resample_noise()
                self.target_net.resample_noise()

            if self.distributional:
                # ---- QR-DQN (AUSBAUPLAN Phase E) ------------------------- #
                # Das Netz liefert je Aktion N Quantil-Stuetzstellen der
                # Ergebnis-Verteilung. Gelernt wird per Quantile-Huber-Loss:
                # jede Stuetzstelle i "zieht" sich an ihr Ziel-Quantil, mit
                # asymmetrischem Gewicht |tau_i - 1{u<0}| -- so faechern sich
                # die N Werte von selbst zur echten Verteilung auf.
                N = self.quantile
                theta = self.policy_net(s)                       # (B, 3, N)
                idx = a.view(-1, 1, 1).expand(-1, 1, N)
                theta_a = theta.gather(1, idx).squeeze(1)        # (B, N)

                with torch.no_grad():
                    if cfg.double_dqn:
                        best_next = self.policy_net.q_values(s2).argmax(dim=1)
                    else:
                        best_next = self.target_net.q_values(s2).argmax(dim=1)
                    idx2 = best_next.view(-1, 1, 1).expand(-1, 1, N)
                    theta_next = self.target_net(s2).gather(1, idx2).squeeze(1)
                    q_target = (r.unsqueeze(1)
                                + disc.unsqueeze(1) * theta_next
                                * (1.0 - d).unsqueeze(1))        # (B, N)

                # u[b, i, j] = Ziel-Quantil j minus Vorhersage-Quantil i.
                u = q_target.unsqueeze(1) - theta_a.unsqueeze(2)  # (B, N, N)
                huber = torch.where(u.abs() <= 1.0,
                                    0.5 * u * u, u.abs() - 0.5)
                tau = ((torch.arange(N, device=self.device, dtype=torch.float32)
                        + 0.5) / N).view(1, N, 1)
                elementwise = ((tau - (u.detach() < 0).float()).abs()
                               * huber).mean(dim=2).sum(dim=1)    # (B,)
                loss = (w * elementwise).mean()
                # PER-Prioritaet: Verschaetzung der MITTELWERTE (kompatibel
                # zum bestehenden update_priorities-Format).
                td_error = q_target.mean(dim=1) - theta_a.detach().mean(dim=1)
            else:
                # ---- klassisches (Double-)DQN, unveraendert -------------- #
                # 1) Was schaetzt das Netz AKTUELL fuer die damals gewaehlte
                #    Aktion? gather() pickt aus den 3 Q-Werten den passenden.
                q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

                # 2) Was WAERE der richtige Wert (Bellman)? -> Target-Netz.
                with torch.no_grad():
                    if cfg.double_dqn:
                        # Double DQN: das lernende Netz sagt, WELCHE Aktion in
                        # der Folgesituation die beste waere, das eingefrorene
                        # Netz sagt, WAS sie wert ist. Zwei getrennte Meinungen
                        # -- das bremst die bekannte Neigung von DQN, sich
                        # selbst zu ueberschaetzen.
                        best_next = self.policy_net(s2).argmax(dim=1, keepdim=True)
                        q_next = self.target_net(s2).gather(1, best_next).squeeze(1)
                    else:
                        q_next = self.target_net(s2).max(dim=1).values

                    # `disc` statt cfg.gamma: bei n-Schritt-Ketten ist die
                    # Belohnung schon ueber mehrere Zuege zusammengefasst, der
                    # Sprung in die Zukunft ist also gamma^n gross (und am
                    # Partie-Ende kuerzer). (1 - d): war die Partie zu Ende,
                    # gibt es KEINE Zukunft mehr.
                    q_target = r + disc * q_next * (1.0 - d)

                # 3) Wie falsch lag das Netz? Huber-/SmoothL1-Loss ist gegen
                #    einzelne Ausreisser robuster als der quadratische Fehler.
                #    `w` gewichtet die Eintraege (priorisiertes Tagebuch).
                td_error = q_target - q_pred
                elementwise = F.smooth_l1_loss(q_pred, q_target, reduction="none")
                loss = (w * elementwise).mean()

            # 4) Regler nachjustieren (Gradientenabstieg).
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                # Gradienten kappen: verhindert einzelne riesige Lernschritte,
                # die das Netz aus der Bahn werfen wuerden.
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), cfg.grad_clip)
            self.optimizer.step()

            self.learn_steps += 1
            last_loss = float(loss.item())

            # 5) Dem Tagebuch zurueckmelden, wie ueberraschend jede Erinnerung
            #    war -- damit die lehrreichen oefter gezogen werden.
            buffer.update_priorities(indices, td_error.detach().cpu().numpy())

            # 6) Ab und zu die Zielscheibe neu aufstellen.
            if self.learn_steps % cfg.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return last_loss

    # ------------------------------------------------------------------ #
    # Vorhandenes Gehirn weiterverwenden
    # ------------------------------------------------------------------ #
    def load_state_dict(self, state_dict) -> None:
        """Uebernimmt gespeicherte Gewichte in BEIDE Netze (Weitertrainieren)."""
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)

    # ------------------------------------------------------------------ #
    # Speichern / Laden
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str, meta: dict) -> str:
        """Speichert das Gehirn + Trainings-Metadaten als .pt-Datei.

        Gleiches Format wie beim Neuroevolution-Champion (siehe
        ai/torch_bridge.py), damit man denselben Zuschau-Weg benutzen kann.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            # Bewusst auf die CPU kopiert: sonst traegt die Datei das
            # Trainings-Geraet in sich (z.B. "mps") und laesst sich auf einem
            # Rechner ohne dieses Geraet nur noch mit map_location oeffnen.
            # So bleibt jeder Champion ueberall lesbar (Mac, Windows-PC).
            "state_dict": {k: v.detach().cpu()
                           for k, v in self.policy_net.state_dict().items()},
            "hidden": tuple(self.cfg.hidden),
            # Ohne diese Angaben wuesste ein Zuschau-Programm spaeter nicht,
            # mit welcher Wahrnehmung/Aktivierung das Netz gefuettert werden
            # will bzw. gebaut werden muss.
            "input_size": self.input_size,
            "perception": self.cfg.perception,
            "activation": self.policy_net.activation,
            # Architektur-Schalter (AUSBAUPLAN.md) -- Ladepfade nehmen bei
            # fehlendem Feld die alten Defaults an (mlp/False/False/0).
            "network": getattr(self.cfg, "network", "mlp"),
            "dueling": getattr(self.cfg, "dueling", False),
            "noisy": getattr(self.cfg, "noisy", False),
            "distributional": getattr(self.cfg, "distributional", False),
            "quantile_anzahl": self.quantile,
            # CNN-Geometrie: ohne die koennte watch_ai/_resume das Netz nicht
            # in der richtigen Groesse nachbauen (state_dict wuerde nicht passen).
            "cnn_kanaele": tuple(getattr(self.cfg, "cnn_kanaele", (16, 32))),
            "cnn_pool": getattr(self.cfg, "cnn_pool", 6),
        }
        payload.update(meta)
        torch.save(payload, path)
        return path
