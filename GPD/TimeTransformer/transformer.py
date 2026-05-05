import math

import torch
import torch.nn as nn
from einops import rearrange

from TimeTransformer.utils import generate_original_PE, generate_regular_PE


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class ConditionedParameterTransformer(nn.Module):
    def __init__(
        self,
        d_input,
        d_model,
        d_output,
        num_wl,
        N=4,
        layernum=0,
        dropout=0.1,
        pe="original",
        pe_period=None,
        wl_embedding_dim=8,
        nhead=8,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        **_,
    ):
        super().__init__()

        self._d_model = d_model
        self.layernum = layernum
        self.channels = d_input
        self.condition_dim = wl_embedding_dim + 2
        self.wl_embedding = nn.Embedding(num_wl, wl_embedding_dim)

        self.input_projection = nn.Linear(d_input + self.condition_dim, d_model)
        self.output_projection = nn.Linear(d_model, d_output)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N)

        pe_functions = {
            "original": generate_original_PE,
            "regular": generate_regular_PE,
        }
        if pe in pe_functions:
            self._generate_PE = pe_functions[pe]
            self._pe_period = pe_period
        elif pe is None:
            self._generate_PE = None
            self._pe_period = None
        else:
            raise NameError('PE "{}" not understood.'.format(pe))

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(d_model)
            fourier_dim = d_model

        self.step_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def _build_condition_tokens(self, condition, sequence_length):
        wl_ids = condition[:, 0].round().long()
        wl_ids = wl_ids.clamp(min=0, max=self.wl_embedding.num_embeddings - 1)
        wl_embedding = self.wl_embedding(wl_ids)
        continuous_condition = condition[:, 1:].float()
        condition_vector = torch.cat((wl_embedding, continuous_condition), dim=-1)
        return condition_vector.unsqueeze(1).expand(-1, sequence_length, -1)

    def forward(self, x, t, condition, x_self_cond=None):
        del x_self_cond

        tokens = x.permute(0, 2, 1)
        sequence_length = tokens.shape[1]
        condition_tokens = self._build_condition_tokens(condition, sequence_length)
        tokens = torch.cat((tokens, condition_tokens), dim=-1)

        encoding = self.input_projection(tokens)
        encoding = encoding + self.step_mlp(t).unsqueeze(1)

        if self._generate_PE is not None:
            pe_kwargs = {"period": self._pe_period} if self._pe_period else {}
            positional_encoding = self._generate_PE(sequence_length, self._d_model, **pe_kwargs).to(encoding.device)
            encoding = encoding + positional_encoding.unsqueeze(0)

        encoding = self.encoder(encoding)
        output = self.output_projection(encoding)
        return output.permute(0, 2, 1)


class Transformer1(ConditionedParameterTransformer):
    pass


class Transformer2(ConditionedParameterTransformer):
    pass


class Transformer3(ConditionedParameterTransformer):
    pass


class Transformer4(ConditionedParameterTransformer):
    pass


class Transformer5(ConditionedParameterTransformer):
    pass
