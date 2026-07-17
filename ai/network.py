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

Diese Datei enthaelt bewusst NUR reines NumPy (kein PyTorch-Import!), damit das
Neuroevolution-Training (ai/evolution/train_evolution.py) auch unter PyPy3
laeuft -- PyPy hat unter Windows keine fertigen PyTorch-Wheels, ein Import hier
wuerde das Training dort sofort crashen lassen.

Die PyTorch-Sicht auf dasselbe Netz (`SnakeNet`, kanonisches Speicherformat,
spaeter auch fuer DQN) lebt getrennt in ai/torch_bridge.py und wird nur dort
gebraucht, wo wirklich Torch noetig ist (Speichern des Champions, Zuschauen).
`NumpyPolicy` hier ist die schlanke, sehr schnelle Vorwaertsrechnung fuer die
Neuroevolution (die wertet zehntausende Partien aus, da zaehlt jede Mikro-
sekunde) -- sie rechnet exakt dasselbe wie SnakeNet (per Test abgesichert).
"""

from __future__ import annotations

import numpy as np

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
