# # save_output.py

# from datasets.viton_dataset import VITONDataset
# from models.full_vton import FullVTON

# import torch
# from torchvision.utils import save_image

# dataset = VITONDataset(
#     root_dir=r"C:\Users\valkontek 010\Downloads\VTON-HD"
# )

# sample = dataset[0]

# person = sample["person"].unsqueeze(0)

# cloth = sample["cloth"].unsqueeze(0)

# pose = sample["pose"].unsqueeze(0)

# model = FullVTON()

# model.eval()

# with torch.no_grad():

#     output = model(
#         person,
#         cloth,
#         pose
#     )

# save_image(
#     output,
#     "output.png"
# )

# print("Saved output.png")

from models.full_vton import FullVTON

model = FullVTON()

total = sum(
    p.numel()
    for p in model.parameters()
)

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    f"Total Parameters: {total:,}"
)

print(
    f"Trainable Parameters: {trainable:,}"
)