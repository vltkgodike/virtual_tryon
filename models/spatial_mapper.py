import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOFeatureMapper(nn.Module):

    def __init__(
        self,
        output_size=18
    ):
        super().__init__()

        self.output_size = output_size

    def forward(self, tokens):

        if tokens.dim() != 3:
            raise ValueError(
                "Expected tokens with shape [batch, tokens, channels]"
            )

        b, n, c = tokens.shape

        spatial_tokens = tokens
        spatial_tokens_count = n

        grid_size = int(
            spatial_tokens_count ** 0.5
        )

        # DINO tokens include a CLS token; pose tokens do not.
        if grid_size * grid_size != spatial_tokens_count:

            spatial_tokens = tokens[:, 1:, :]
            spatial_tokens_count = spatial_tokens.shape[1]

            grid_size = int(
                spatial_tokens_count ** 0.5
            )

        if grid_size * grid_size != spatial_tokens_count:
            raise ValueError(
                f"Cannot reshape {n} tokens into a square feature map"
            )

        x = spatial_tokens.reshape(
            b,
            grid_size,
            grid_size,
            c
        )

        x = x.permute(
            0,
            3,
            1,
            2
        )

        if (
            self.output_size is not None
            and x.shape[-2:] != (
                self.output_size,
                self.output_size
            )
        ):
            x = F.interpolate(
                x,
                size=(
                    self.output_size,
                    self.output_size
                ),
                mode="bilinear",
                align_corners=False
            )

        return x
    
