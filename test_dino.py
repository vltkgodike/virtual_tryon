from datasets.viton_dataset import (
    VITONDataset
)

from models.dino_encoder import (
    DINOEncoder
)

from models.pose_encoder import (
    PoseEncoder
)

from models.multimodal_uwt import (
    MultiModalUWT
)

import torch


dataset = VITONDataset(
    root_dir=r"C:\Users\valkontek 010\Downloads\VTON-HD"
)

sample = dataset[0]

person = sample["person"].unsqueeze(0)

cloth = sample["cloth"].unsqueeze(0)

pose = sample["pose"].unsqueeze(0)

dino = DINOEncoder()

pose_encoder = PoseEncoder()

uwt = MultiModalUWT()

with torch.no_grad():

    person_tokens = dino(
        person
    )

    cloth_tokens = dino(
        cloth
    )

    pose_tokens = pose_encoder(
        pose
    )

    fused_tokens = uwt(
        person_tokens,
        cloth_tokens,
        pose_tokens
    )

print()

print(
    "Person Tokens:",
    person_tokens.shape
)

print(
    "Cloth Tokens:",
    cloth_tokens.shape
)

print(
    "Pose Tokens:",
    pose_tokens.shape
)

print()

print(
    "Final Fused Tokens:"
)

print(
    fused_tokens.shape
)
from models.token_compressor import (
    TokenCompressor
)

compressor = TokenCompressor()

compressed = compressor(
    fused_tokens
)

print("fused tokens shape:")

print(
    compressed.shape
)
from models.token_to_feature import (
    TokenToFeatureMap
)

mapper = TokenToFeatureMap()

feature_map = mapper(
    compressed
)
print("feature map shape:")
print(
    feature_map.shape
)

from models.decoder import Decoder

decoder = Decoder()

output = decoder(
    feature_map
)
print("output shape:")
print(
    output.shape
)

from datasets.viton_dataset import (
    VITONDataset
)

from models.full_vton import (
    FullVTON
)

import torch


dataset = VITONDataset(
    root_dir=r"C:\Users\valkontek 010\Downloads\VTON-HD"
)

sample = dataset[0]

person = sample["person"].unsqueeze(0)

cloth = sample["cloth"].unsqueeze(0)

pose = sample["pose"].unsqueeze(0)

model = FullVTON()

with torch.no_grad():

    output = model(
        person,
        cloth,
        pose
    )

print("Output shape:")

print(
    output.shape
)