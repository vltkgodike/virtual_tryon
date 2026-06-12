import torch
import torch.nn as nn

from models.dino_encoder import DINOEncoder
from models.pose_encoder import PoseEncoder
from models.multimodal_uwt import MultiModalUWT
from models.token_compressor import TokenCompressor
from models.token_to_feature import TokenToFeatureMap
from models.decoder import Decoder


class FullVTON(nn.Module):

    def __init__(self):

        super().__init__()

        self.dino = DINOEncoder()
        # Freeze DINOv2
        for param in self.dino.parameters():
            param.requires_grad = False
        self.pose_encoder = PoseEncoder()

        self.uwt = MultiModalUWT()

        self.compressor = TokenCompressor()

        self.mapper = TokenToFeatureMap()

        self.decoder = Decoder()

    def forward(
        self,
        person,
        cloth,
        pose
    ):

        person_tokens = self.dino(
            person
        )

        cloth_tokens = self.dino(
            cloth
        )

        pose_tokens = self.pose_encoder(
            pose
        )

        fused_tokens = self.uwt(
            person_tokens,
            cloth_tokens,
            pose_tokens
        )

        compressed = self.compressor(
            fused_tokens
        )

        feature_map = self.mapper(
            compressed
        )

        output = self.decoder(
            feature_map
        )

        return output