import torch
import torch.nn as nn

class TokenCompressor(nn.Module):

    def __init__(
        self,
        output_tokens=256
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(
            output_tokens
        )

    def forward(
        self,
        tokens
    ):

        # [B,N,C]
        tokens = tokens.transpose(
            1,
            2
        )

        # [B,C,N]

        tokens = self.pool(
            tokens
        )

        # [B,C,256]

        tokens = tokens.transpose(
            1,
            2
        )

        # [B,256,C]

        return tokens