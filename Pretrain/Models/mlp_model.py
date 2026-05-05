import torch.nn as nn


try:
    from .meta_module import MetaModule  # type: ignore
except ImportError:
    MetaModule = nn.Module


class MLPRegressor(MetaModule):
    def __init__(self, in_dim=4, hidden_dims=(64, 32), out_dim=1):
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, out_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.network(x)
