import torch
import torch.nn as nn


class Decoder(
    nn.Module
):

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256,
                128,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                16,
                4,
                2,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                16,
                3,
                3,
                1,
                1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        x
    ):

        return self.decoder(
            x
        )
    
