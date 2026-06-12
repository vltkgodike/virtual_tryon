import torch
import torch.nn as nn


class TokenMapper(
    nn.Module
):

    def __init__(
        self,
        in_dim=768,
        out_dim=256
    ):

        super().__init__()

        self.proj = nn.Linear(
            in_dim,
            out_dim
        )

    def forward(
        self,
        tokens
    ):

        # remove CLS token

        tokens = tokens[:, 1:, :]

        b, n, c = tokens.shape

        tokens = self.proj(
            tokens
        )

        feature_map = tokens.reshape(
            b,
            16,
            16,
            256
        )

        feature_map = (
            feature_map
            .permute(0,3,1,2)
        )

        return feature_map