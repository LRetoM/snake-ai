"""Das neuronale Netz -- das "Gehirn" der KI. Von BEIDEN KIs geteilt.

Ein neuronales Netz ist hier nichts Magisches: Es nimmt die 11 Wahrnehmungszahlen
(aus perception.py) und rechnet daraus 3 Ausgabezahlen -- eine "Lust" pro Aktion
(geradeaus / links / rechts). Die hoechste gewinnt und wird der Zug.

Analogie: ein Mischpult mit vielen Reglern (den "Gewichten"). Am Anfang stehen
alle Regler zufaellig -> die Schlange f"aehrt Bloedsinn. "Lernen" heisst nur: die
Regler so verstellen, dass gute Zuege rauskommen. WIE man sie verstellt, ist der
Unterschied zwischen den beiden KIs:
- Neuroevolution (zuerst): durch Zucht/Auswahl ueber Generationen (kein Rechnen
  am einzelnen Netz). Dafuer behandeln wir alle Gewichte als einen flachen
  Zahlenvektor -- das "Genom".
- DQN (spaeter): durch gezieltes Nachrechnen (Gradienten) mit PyTorch.

Deshalb bietet diese Datei ZWEI Sichten auf dasselbe Netz:
1. `SnakeNet` -- ein PyTorch-Modul. Das ist das kanonische/gespeicherte Format
   (torch.save), und DQN wird spaeter genau damit rechnen.
2. `NumpyPolicy` -- eine schlanke, sehr schnelle Vorwaertsrechnung in reinem NumPy
   fuer die Neuroevolution (die wertet zehntausende Partien aus, da zaehlt jede
   Mikrosekunde). Sie benutzt exakt dieselbe Rechnung wie SnakeNet.

Ein Test stellt sicher, dass beide Sichten identische Ergebnisse liefern.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ai.perception import INPUT_SIZE

OUTPUT_SIZE = 3  # geradeaus / links / rechts
DEFAULT_HIDDEN: tuple[int, ...] = (20, 12)  # zwei versteckte Schichten


# --------------------------------------------------------------------------- #
# Architektur-Helfer (unabhaengig von PyTorch)
# --------------------------------------------------------------------------- #
def layer_shapes(hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> list[tuple[int, int]]:
    """Liefert die (in, out)-Groessen aller Schichten der Reihe nach.

    Beispiel hidden=(20,12): 11->20, 20->12, 12->3.
    """
    sizes = [INPUT_SIZE, *hidden, OUTPUT_SIZE]
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def genome_size(hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> int:
    """Anzahl aller Gewichte + Bias-Werte = Laenge des Genom-Vektors."""
    # pro Schicht: (in*out) Gewichte + (out) Bias-Werte
    return sum(inp * out + out for inp, out in layer_shapes(hidden))


def random_genome(hidden: tuple[int, ...] = DEFAULT_HIDDEN,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Ein zufaelliges Genom (frisch initialisiertes Netz).

    Gewichte werden mit kleiner Streuung ~ 1/sqrt(in) gezogen (uebliche
    Initialisierung, damit die Werte anfangs nicht explodieren), Bias = 0.
    """
    rng = rng or np.random.default_rng()
    parts = []
    for inp, out in layer_shapes(hidden):
        scale = 1.0 / np.sqrt(inp)
        parts.append(rng.normal(0.0, scale, size=inp * out).astype(np.float32))  # Gewichte
        parts.append(np.zeros(out, dtype=np.float32))                            # Bias
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Schnelle NumPy-Vorwaertsrechnung (fuer die Neuroevolution)
# --------------------------------------------------------------------------- #
class NumpyPolicy:
    """Wertet ein Genom sehr schnell aus: Wahrnehmung (11) -> Aktion (0/1/2).

    Zerlegt den flachen Genom-Vektor einmal in Gewichtsmatrizen + Bias und rechnet
    dann pro Zug nur zwei/drei kleine Matrixmultiplikationen. Kein PyTorch-Overhead.
    Rechnet exakt wie PyTorchs nn.Linear: y = x @ W.T + b (W hat Form (out, in)).
    """

    def __init__(self, genome: np.ndarray, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> None:
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        idx = 0
        for inp, out in layer_shapes(hidden):
            w = genome[idx:idx + inp * out].reshape(out, inp)  # (out, in) wie torch
            idx += inp * out
            b = genome[idx:idx + out]
            idx += out
            self.layers.append((w, b))

    def act(self, observation: np.ndarray) -> int:
        """Gibt die gewaehlte Aktion zurueck (Index der groessten Ausgabe)."""
        x = observation
        last = len(self.layers) - 1
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < last:
                x = np.tanh(x)  # Aktivierung in den versteckten Schichten
        return int(np.argmax(x))


# --------------------------------------------------------------------------- #
# PyTorch-Modul (kanonisches Format zum Speichern + spaeter fuer DQN)
# --------------------------------------------------------------------------- #
class SnakeNet(nn.Module):
    """Dasselbe Netz als PyTorch-Modul: 11 -> hidden... -> 3, tanh in den Hidden."""

    def __init__(self, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.hidden = tuple(hidden)
        layers = []
        prev = INPUT_SIZE
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            prev = h
        self.hidden_layers = nn.ModuleList(layers)
        self.out = nn.Linear(prev, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self.out(x)

    # -- Genom <-> Netzgewichte (gleiche Reihenfolge wie NumpyPolicy) --------- #
    def get_genome(self) -> np.ndarray:
        """Alle Gewichte + Bias als einen flachen float32-Vektor."""
        parts = []
        for layer in list(self.hidden_layers) + [self.out]:
            parts.append(layer.weight.detach().cpu().numpy().reshape(-1))
            parts.append(layer.bias.detach().cpu().numpy().reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def load_genome(self, genome: np.ndarray) -> None:
        """Laedt Gewichte + Bias aus einem flachen Vektor ins Netz."""
        idx = 0
        with torch.no_grad():
            for layer in list(self.hidden_layers) + [self.out]:
                out_f, in_f = layer.weight.shape
                w = genome[idx:idx + out_f * in_f].reshape(out_f, in_f)
                idx += out_f * in_f
                b = genome[idx:idx + out_f]
                idx += out_f
                layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)))
                layer.bias.copy_(torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32)))


def genome_to_net(genome: np.ndarray, hidden: tuple[int, ...] = DEFAULT_HIDDEN) -> SnakeNet:
    """Baut aus einem Genom ein PyTorch-Netz (z.B. zum Speichern des Champions)."""
    net = SnakeNet(hidden)
    net.load_genome(genome)
    return net
