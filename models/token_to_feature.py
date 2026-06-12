import torch
import torch.nn as nn


class TokenToFeatureMap(nn.Module):

    def __init__(
        self,
        embed_dim=768,
        out_channels=256
    ):
        super().__init__()

        self.proj = nn.Linear(
            embed_dim,
            out_channels
        )

    def forward(
        self,
        tokens
    ):

        x = self.proj(tokens)

        b, n, c = x.shape

        x = x.reshape(
            b,
            16,
            16,
            c
        )

        x = x.permute(
            0,
            3,
            1,
            2
        )

        return x