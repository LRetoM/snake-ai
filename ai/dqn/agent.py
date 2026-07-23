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
from ai.torch_bridge import SnakeNet


def pick_device() -> torch.device:
    """CUDA-Grafikkarte, falls vorhanden -- sonst CPU.

    MPS (Apple-Grafik) lassen wir bewusst weg: unser Netz ist winzig, und der
    Weg zur Grafikkarte und zurueck kostet mehr Zeit, als er einspart. Bei
    11 Eingaengen ist die CPU schlicht schneller.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
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
        self.device = pick_device()
        self.rng = np.random.default_rng(seed)

        # Wie viele CPU-Threads soll PyTorch benutzen? Das ist der mit Abstand
        # groesste Geschwindigkeits-Hebel und je nach Rechner voellig
        # unterschiedlich: mal ist 1 Thread am schnellsten (kleine Netze, die
        # Abstimmung zwischen Threads kostet mehr als sie bringt), mal 4.
        # Deshalb wird es bei torch_threads=0 einmalig GEMESSEN statt geraten.
        if cfg.torch_threads:
            torch.set_num_threads(int(cfg.torch_threads))
        else:
            self.threads = tune_threads(input_size, cfg.hidden, cfg.batch_size)
        self.threads = torch.get_num_threads()

        # relu statt tanh: tanh "saettigt" bei den fuer DQN typischen groesseren
        # Q-Werten (Steigung geht gegen 0, Lernsignal versickert) -- relu hat
        # dieses Problem nicht. Ueber cfg.activation einstellbar/messbar.
        activation = getattr(cfg, "activation", "tanh")

        # policy_net = das Netz, das entscheidet UND trainiert wird.
        self.policy_net = SnakeNet(cfg.hidden, input_size, activation).to(self.device)
        # target_net = die eingefrorene Kopie, die nur das Lernziel liefert.
        self.target_net = SnakeNet(cfg.hidden, input_size, activation).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # wird nie direkt trainiert

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
        tensor = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
        q_values = self.policy_net(tensor.to(self.device))

        # Bester bekannter Zug pro Spiel.
        actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)
        self.last_mean_q = float(q_values.max(dim=1).values.mean().item())

        # ... und jetzt die Neugier: bei manchen Spielen stattdessen wuerfeln.
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

            # 1) Was schaetzt das Netz AKTUELL fuer die damals gewaehlte Aktion?
            #    gather() pickt aus den 3 Q-Werten genau den zur Aktion passenden.
            q_pred = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # 2) Was WAERE der richtige Wert (Bellman)? -> mit dem Target-Netz.
            with torch.no_grad():
                if cfg.double_dqn:
                    # Double DQN: das lernende Netz sagt, WELCHE Aktion in der
                    # Folgesituation die beste waere, das eingefrorene Netz sagt,
                    # WAS sie wert ist. Zwei getrennte Meinungen -- das bremst die
                    # bekannte Neigung von DQN, sich selbst zu ueberschaetzen
                    # ("ich haette da bestimmt 20 Punkte geholt").
                    best_next = self.policy_net(s2).argmax(dim=1, keepdim=True)
                    q_next = self.target_net(s2).gather(1, best_next).squeeze(1)
                else:
                    q_next = self.target_net(s2).max(dim=1).values

                # `disc` statt cfg.gamma: bei n-Schritt-Ketten ist die Belohnung
                # schon ueber mehrere Zuege zusammengefasst, der Sprung in die
                # Zukunft ist also gamma^n gross (und am Partie-Ende kuerzer).
                # (1 - d): war die Partie zu Ende, gibt es KEINE Zukunft mehr --
                # dann ist der wahre Wert einfach die Belohnung selbst.
                q_target = r + disc * q_next * (1.0 - d)

            # 3) Wie falsch lag das Netz? Huber-/SmoothL1-Loss ist gegenueber
            #    einzelnen Ausreissern robuster als der quadratische Fehler --
            #    ein ungluecklicher -10-Ausreisser reisst das Training nicht um.
            #    `w` gewichtet die Eintraege: beim priorisierten Tagebuch zaehlen
            #    haeufig gezogene Erinnerungen entsprechend weniger (sonst waere
            #    das Weltbild der KI verzerrt, siehe memory.py).
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
            "state_dict": self.policy_net.state_dict(),
            "hidden": tuple(self.cfg.hidden),
            # Ohne diese Angaben wuesste ein Zuschau-Programm spaeter nicht,
            # mit welcher Wahrnehmung/Aktivierung das Netz gefuettert werden
            # will bzw. gebaut werden muss.
            "input_size": self.input_size,
            "perception": self.cfg.perception,
            "activation": self.policy_net.activation,
        }
        payload.update(meta)
        torch.save(payload, path)
        return path
