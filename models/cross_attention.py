import torch
import torch.nn as nn


class CrossAttentionBlock(
    nn.Module
):

    def __init__(
        self,
        embed_dim=768,
        num_heads=8
    ):

        super().__init__()

        self.cross_attn = (
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True
            )
        )

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.ffn = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim * 4
            ),

            nn.GELU(),

            nn.Linear(
                embed_dim * 4,
                embed_dim
            )
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

    def forward(
        self,
        person_tokens,
        cloth_tokens
    ):

        attn_output, attn_weights = (
            self.cross_attn(
                query=person_tokens,
                key=cloth_tokens,
                value=cloth_tokens
            )
        )

        x = self.norm1(
            person_tokens + attn_output
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x, attn_weights