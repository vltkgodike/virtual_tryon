import torch
import torch.nn as nn


class PoseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )

        self.token_projection = nn.Linear(
            256,
            768
        )

    def forward(
        self,
        pose
    ):

        x = self.encoder(
            pose
        )

        b, c, h, w = x.shape

        x = x.flatten(2)

        x = x.transpose(
            1,
            2
        )

        x = self.token_projection(
            x
        )

        return x