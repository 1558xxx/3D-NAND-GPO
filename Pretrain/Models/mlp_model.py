import torch.nn as nn


try:
    from .meta_module import MetaModule  # type: ignore
except ImportError:
    MetaModule = nn.Module


def _build_activation(name):
    activation_name = str(name).lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "silu":
        return nn.SiLU()
    raise ValueError("Unsupported activation: {}".format(name))


class MLPRegressor(MetaModule):
    def __init__(self, in_dim=4, hidden_dims=(64, 32), out_dim=1, activation="relu"):
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        act_1 = _build_activation(activation)
        act_2 = _build_activation(activation)
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_1),
            act_1,
            nn.Linear(hidden_1, hidden_2),
            act_2,
            nn.Linear(hidden_2, out_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.network(x)
