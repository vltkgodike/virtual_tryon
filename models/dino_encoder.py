from transformers import (
    AutoImageProcessor,
    Dinov2Model
)

import torch
import torch.nn as nn


class DINOEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.processor = (
            AutoImageProcessor
            .from_pretrained(
                "facebook/dinov2-base"
            )
        )

        self.model = (
            Dinov2Model
            .from_pretrained(
                "facebook/dinov2-base"
            )
        )

    def forward(
        self,
        pixel_values
    ):

        outputs = self.model(
            pixel_values=pixel_values
        )

        return outputs.last_hidden_state