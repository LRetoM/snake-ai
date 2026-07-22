"""Die PyTorch-Sicht auf das Netz -- getrennt von ai/network.py, damit das
Neuroevolution-Training selbst (ai/evolution/train_evolution.py) OHNE PyTorch
auskommt (reines NumPy). Torch wird erst hier gebraucht: beim Speichern des
Champions als .pt-Datei und beim Zuschauen (watch_ai.py), und spaeter fuer DQN.

Grund fuer die Trennung: PyPy3 (JIT-beschleunigtes Python, 5-10x schneller
fuers Training) hat unter Windows keine fertigen PyTorch-Wheels -- ein Import
von torch wuerde das Training dort sofort crashen lassen. Da das Training
selbst nur die schlanke NumpyPolicy (ai/network.py) braucht, bleibt der
Trainings-Pfad frei von diesem Import.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from ai.network import DEFAULT_HIDDEN, OUTPUT_SIZE
from ai.perception import INPUT_SIZE


class SnakeNet(nn.Module):
    """Dasselbe Netz als PyTorch-Modul: N -> hidden... -> 3, tanh in den Hidden.

    `input_size` ist normalerweise 11 (die Standard-Wahrnehmung, die auch die
    Neuroevolution benutzt). Das DQN kann optional mit einer REICHEREN
    Wahrnehmung laufen (ai/perception.py, `perceive_rich`) -- dafuer muss die
    Eingangsschicht groesser sein. Deshalb ist die Groesse hier einstellbar,
    mit dem alten Wert als Standard: bestehende Checkpoints laden unveraendert.
    """

    def __init__(self, hidden: tuple[int, ...] = DEFAULT_HIDDEN,
                 input_size: int = INPUT_SIZE) -> None:
        super().__init__()
        self.hidden = tuple(hidden)
        self.input_size = int(input_size)
        layers = []
        prev = self.input_size
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


def save_champion_checkpoint(genome: np.ndarray, meta: dict, model_dir: str) -> str:
    """Baut aus Genom + Metadaten die evo_champion.pt (Format das watch_ai.py
    und das Dashboard erwarten). Gemeinsam genutzt von train_evolution.py
    (wenn Torch direkt verfuegbar ist) und build_champion.py (Nachtraeglicher
    Konvertierungsschritt, wenn unter PyPy ohne Torch trainiert wurde)."""
    net = genome_to_net(genome, tuple(meta["hidden"]))
    path = os.path.join(model_dir, "evo_champion.pt")
    torch.save(
        {
            "state_dict": net.state_dict(),
            "hidden": tuple(meta["hidden"]),
            "score": meta["score"],
            "generation": meta["generation"],
            "grid_cols": meta["grid_cols"],
            "grid_rows": meta["grid_rows"],
            "fruit_count": meta["fruit_count"],
            "wrap_walls": meta["wrap_walls"],
            "episodes_per_genome": meta["episodes_per_genome"],
        },
        path,
    )
    return path
