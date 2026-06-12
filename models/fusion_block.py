import torch
import torch.nn as nn


class FusionBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.fusion = nn.Sequential(

            nn.Conv2d(
                768 * 3,
                512,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                512,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

    def forward(
        self,
        person_map,
        cloth_map,
        pose_map
    ):

        x = torch.cat(
            [
                person_map,
                cloth_map,
                pose_map
            ],
            dim=1
        )

        return self.fusion(x)