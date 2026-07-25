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


class NoisyLinear(nn.Module):
    """Linear-Schicht mit LERNBAREM Rauschen (Noisy Nets, AUSBAUPLAN Phase C).

    Idee: Statt pro Zug zu WUERFELN (epsilon-greedy) sitzt die Erkundung als
    Rauschen direkt IN den Gewichten -- jedes Gewicht ist mu + sigma*eps mit
    lernbarem sigma. Wo das Netz sicher ist, lernt es sigma gegen 0 (dort
    wird es von selbst deterministisch); wo es unsicher ist, bleibt Rauschen
    und damit Erkundung. Der Vorteil fuer uns: kein einzelner Wuerfel-Zug
    toetet mehr eine 300-Zuege-Partie.

    "Factorised Gaussian" (Standard aus dem Rainbow-Paper): statt out*in
    unabhaengiger Rauschwerte nur out+in, per f(x)=sign(x)*sqrt(|x|)
    kombiniert -- deutlich billiger, praktisch gleich gut.

    Im eval()-Modus rechnet die Schicht NUR mit mu (rauschfrei) -- dadurch
    sind Pruefungen und Zuschauen automatisch deterministisch-ehrlich.
    """

    def __init__(self, in_features: int, out_features: int,
                 sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / (in_features ** 0.5)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma_init * bound))
        self.bias_mu = nn.Parameter(
            torch.empty(out_features).uniform_(-bound, bound))
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma_init * bound))
        self.register_buffer("weight_eps", torch.zeros(out_features, in_features))
        self.register_buffer("bias_eps", torch.zeros(out_features))
        self.resample_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def resample_noise(self) -> None:
        """Zieht frisches Rauschen (einmal pro Entscheidung/Lernschritt)."""
        eps_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        eps_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_eps = eps_out.outer(eps_in)
        self.bias_eps = eps_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu + self.bias_sigma * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu   # eval = rauschfrei
        return nn.functional.linear(x, w, b)


def _linear(noisy: bool, in_f: int, out_f: int) -> nn.Module:
    """Baut Linear oder NoisyLinear -- je nach Noisy-Nets-Schalter."""
    return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)


class SnakeNet(nn.Module):
    """Dasselbe Netz als PyTorch-Modul: N -> hidden... -> 3 Q-Werte.

    `input_size` ist normalerweise 11 (die Standard-Wahrnehmung, die auch die
    Neuroevolution benutzt). Das DQN kann optional mit einer REICHEREN
    Wahrnehmung laufen (ai/perception.py, `perceive_rich`) -- dafuer muss die
    Eingangsschicht groesser sein. Deshalb ist die Groesse hier einstellbar,
    mit dem alten Wert als Standard: bestehende Checkpoints laden unveraendert.

    `activation` (neu): "tanh" oder "relu" in den versteckten Schichten.
    Default bleibt "tanh" -- WICHTIG fuer die Neuroevolution, deren Mutation/
    Crossover auf begrenzten Gewichten rechnet und die deshalb IMMER tanh
    benutzt (ruft `SnakeNet` ohne dieses Argument auf, siehe genome_to_net
    unten). Das DQN uebergibt explizit "relu" (siehe ai/dqn/config.py):
    tanh "saettigt" bei grossen Werten (die Steigung geht gegen 0, das
    Lernsignal versickert) -- ein Problem, das bei DQN-Q-Werten im Bereich
    von 30-40 tatsaechlich auftritt, bei den kleinen Neuroevolution-Gewichten
    aber nicht relevant ist.

    AUSBAUPLAN-Erweiterungen (alle Default AUS = exakt altes Verhalten;
    die Neuroevolution ruft weiter ohne diese Argumente auf):
    - `dueling` (Phase B): getrennte Koepfe fuer "wie gut ist die Lage"
      (Value, 1 Wert) und "wie viel besser ist Aktion X als der Schnitt"
      (Advantage, 3 Werte); Q = V + A - mean(A).
    - `noisy` (Phase C): alle Schichten werden NoisyLinear (siehe oben),
      Erkundung sitzt dann im Netz statt im Epsilon-Wuerfel.
    - `quantile` (Phase E, QR-DQN): >0 heisst, der Kopf gibt je Aktion
      `quantile` Verteilungs-Stuetzstellen aus statt einem Mittelwert --
      forward liefert dann (batch, 3, quantile). `q_values()` liefert in
      JEDEM Modus (batch, 3) und ist die eine Schnittstelle fuer alle
      "welche Aktion ist die beste"-Konsumenten.
    """

    def __init__(self, hidden: tuple[int, ...] = DEFAULT_HIDDEN,
                 input_size: int = INPUT_SIZE, activation: str = "tanh",
                 dueling: bool = False, noisy: bool = False,
                 quantile: int = 0) -> None:
        super().__init__()
        self.hidden = tuple(hidden)
        self.input_size = int(input_size)
        if activation not in ("tanh", "relu"):
            raise ValueError(f"Unbekannte Aktivierung '{activation}'. Moeglich: tanh, relu")
        self.activation = activation
        self._act_fn = torch.tanh if activation == "tanh" else torch.relu
        self.dueling = bool(dueling)
        self.noisy = bool(noisy)
        self.quantile = int(quantile)

        layers = []
        prev = self.input_size
        for h in hidden:
            layers.append(_linear(self.noisy, prev, h))
            prev = h
        self.hidden_layers = nn.ModuleList(layers)

        # Kopf: je Aktion 1 Wert (klassisch) oder `quantile` Stuetzstellen.
        je_aktion = max(1, self.quantile)
        if self.dueling:
            self.value_head = _linear(self.noisy, prev, je_aktion)
            self.adv_head = _linear(self.noisy, prev, OUTPUT_SIZE * je_aktion)
        else:
            self.out = _linear(self.noisy, prev, OUTPUT_SIZE * je_aktion)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = self._act_fn(layer(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._features(x)
        je_aktion = max(1, self.quantile)
        if self.dueling:
            v = self.value_head(h).view(-1, 1, je_aktion)
            a = self.adv_head(h).view(-1, OUTPUT_SIZE, je_aktion)
            q = v + a - a.mean(dim=1, keepdim=True)
        else:
            q = self.out(h).view(-1, OUTPUT_SIZE, je_aktion)
        if self.quantile:
            return q                       # (batch, 3, quantile)
        return q.squeeze(-1)               # (batch, 3) -- exakt wie frueher

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """IMMER (batch, 3) -- bei Verteilungs-Koepfen das Quantil-Mittel.
        Die eine Schnittstelle fuer act_batch/Pruefung/Zuschauen, damit deren
        Code fuer alle Netz-Varianten identisch bleibt."""
        out = self.forward(x)
        if self.quantile:
            return out.mean(dim=2)
        return out

    def resample_noise(self) -> None:
        """Frisches Rauschen fuer alle NoisyLinear-Schichten (Phase C)."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.resample_noise()

    # -- Genom <-> Netzgewichte (gleiche Reihenfolge wie NumpyPolicy) --------- #
    def _genom_pruefen(self) -> None:
        """Dueling/Noisy/Quantile-Netze haben ein ANDERES Schicht-Layout als
        das Neuroevolution-Genom -- ein stilles falsches Mapping waere fatal,
        deshalb harter Fehler statt Raten."""
        if self.dueling or self.noisy or self.quantile:
            raise RuntimeError(
                "Genom-Roundtrip gibt es nur fuer das klassische Netz -- "
                "dueling/noisy/quantile-Netze haben kein Neuroevolution-"
                "Genom-Layout.")

    def get_genome(self) -> np.ndarray:
        """Alle Gewichte + Bias als einen flachen float32-Vektor."""
        self._genom_pruefen()
        parts = []
        for layer in list(self.hidden_layers) + [self.out]:
            parts.append(layer.weight.detach().cpu().numpy().reshape(-1))
            parts.append(layer.bias.detach().cpu().numpy().reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    def load_genome(self, genome: np.ndarray) -> None:
        """Laedt Gewichte + Bias aus einem flachen Vektor ins Netz."""
        self._genom_pruefen()
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


class SnakeConvNet(nn.Module):
    """CNN-Netz (AUSBAUPLAN Phase D): sieht das GANZE Brett als Bild.

    Eingabe ist derselbe FLACHE Vektor wie bei allen anderen Wahrnehmungen
    (damit ReplayBuffer/n-Schritt/Spiegelung unveraendert weiterlaufen) --
    Layout laut ai/perception.py "cnn_board":
        [6 Skalare] + [Kanal Koerper] + [Kanal Kopf] + [Kanal Frucht]
    Das Netz formt die drei Kanaele intern zu (3, rows, cols) um und schaut
    mit Faltungs-Schichten drauf -- wie ein Auge, das raeumliche MUSTER
    (Spiralen, Sackgassen) als Form erkennt, statt nur Zahlen-Merkmale zu
    lesen. Genau die Faehigkeit, die den rich_grid*-Wahrnehmungen fehlt
    (deren Fenster endet, die Falle schliesst sich oft ausserhalb).

    AdaptiveAvgPool auf ein festes `pool` x `pool`-Raster macht das Netz
    BRETTGROESSEN-TOLERANT: dieselben Gewichte laufen auf 9x9 wie auf 17x15
    (Klein-Feld-Curriculum mit Gewichts-Transfer). Kompromiss: kostet etwas
    Ortsaufloesung -- ein spaeteres A/B darf eine Variante ohne Pooling
    (fixes Brett) testen.

    GESCHWINDIGKEIT (gemessen 2026-07-25, M-Mac, 4 Threads): Faltung ist auf
    der CPU teuer -- mit (32,64) Kanaelen und 11x11-Pool nur ~125 Zuege/s
    gegen ~7400 beim MLP, praktisch komplett im Lernschritt (Batch 512).
    Deshalb sind `kanaele`/`pool` einstellbar (ai/dqn/config.py:
    cnn_kanaele/cnn_pool) und die Standardwerte bewusst schlank -- sonst
    koennte das CNN einen A/B bei gleicher WANDUHR-Zeit nie gewinnen, egal
    wie gut es pro Zug lernt.

    Leitplanke: die Kanaele sind ROHE Sicht (Koerper/Kopf/Frucht je 0/1),
    KEIN Flood-Fill, KEIN "sicherer Weg"-Kanal.
    """

    KANAELE = 3

    def __init__(self, cols: int, rows: int, skalare: int = 6,
                 activation: str = "relu", dueling: bool = False,
                 noisy: bool = False, quantile: int = 0,
                 kanaele: tuple[int, ...] = (16, 32), pool: int = 6) -> None:
        super().__init__()
        self.cols, self.rows = int(cols), int(rows)
        self.skalare = int(skalare)
        self.input_size = self.skalare + self.KANAELE * self.cols * self.rows
        if activation not in ("tanh", "relu"):
            raise ValueError(f"Unbekannte Aktivierung '{activation}'. Moeglich: tanh, relu")
        self.activation = activation
        self._act_fn = torch.tanh if activation == "tanh" else torch.relu
        self.dueling = bool(dueling)
        self.noisy = bool(noisy)
        self.quantile = int(quantile)
        self.kanaele = tuple(kanaele)
        self.pool_groesse = int(pool)

        convs = []
        prev = self.KANAELE
        for k in self.kanaele:
            convs.append(nn.Conv2d(prev, k, 3, padding=1))
            prev = k
        self.convs = nn.ModuleList(convs)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_groesse, self.pool_groesse))
        flach = prev * self.pool_groesse * self.pool_groesse
        # Nur die FC-Schichten werden noisy (Rainbow-Standard) -- Rauschen in
        # den Conv-Filtern braeuchte viel mehr Parameter und bringt wenig.
        self.fc = _linear(self.noisy, flach + self.skalare, 256)
        je_aktion = max(1, self.quantile)
        if self.dueling:
            self.value_head = _linear(self.noisy, 256, je_aktion)
            self.adv_head = _linear(self.noisy, 256, OUTPUT_SIZE * je_aktion)
        else:
            self.out = _linear(self.noisy, 256, OUTPUT_SIZE * je_aktion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skalare = x[:, :self.skalare]
        brett = x[:, self.skalare:].view(-1, self.KANAELE, self.rows, self.cols)
        h = brett
        for conv in self.convs:
            h = self._act_fn(conv(h))
        h = self.pool(h).flatten(1)
        h = self._act_fn(self.fc(torch.cat([h, skalare], dim=1)))
        je_aktion = max(1, self.quantile)
        if self.dueling:
            v = self.value_head(h).view(-1, 1, je_aktion)
            a = self.adv_head(h).view(-1, OUTPUT_SIZE, je_aktion)
            q = v + a - a.mean(dim=1, keepdim=True)
        else:
            q = self.out(h).view(-1, OUTPUT_SIZE, je_aktion)
        if self.quantile:
            return q
        return q.squeeze(-1)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Wie SnakeNet.q_values: immer (batch, 3)."""
        out = self.forward(x)
        if self.quantile:
            return out.mean(dim=2)
        return out

    def resample_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.resample_noise()


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
