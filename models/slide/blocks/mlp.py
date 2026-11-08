import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        f_in: int,
        f_out: int,
        hidden_dim: int = 128,
        hidden_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        act = nn.ReLU()
        layers = [nn.Linear(f_in, hidden_dim), act, nn.Dropout(dropout)]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act, nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_dim, f_out)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
