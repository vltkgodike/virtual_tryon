import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from datasets.viton_dataset import VITONDataset
from models.full_vton import FullVTON


# --------------------------
# CONFIG
# --------------------------

ROOT_DIR = r"/kaggle/input/datasets/adarshsingh0903/virtual-tryon-dataset/Virtual tryon data/"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 5

BATCH_SIZE = 1

LR = 1e-4


# --------------------------
# DATASET
# --------------------------

dataset = VITONDataset(
    root_dir=ROOT_DIR,
    image_size=256
)

dataset = Subset(
    dataset,
    range(10)
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(
    f"Dataset Size: {len(dataset)}"
)


# --------------------------
# MODEL
# --------------------------

model = FullVTON()

model = model.to(
    DEVICE
)

print(
    f"Using Device: {DEVICE}"
)


# --------------------------
# LOSS
# --------------------------

criterion = nn.L1Loss()


# --------------------------
# OPTIMIZER
# --------------------------

optimizer = torch.optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LR
)


# --------------------------
# TRAINING
# --------------------------

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for batch in loader:

        person = batch["person"].to(
            DEVICE
        )

        cloth = batch["cloth"].to(
            DEVICE
        )

        pose = batch["pose"].to(
            DEVICE
        )

        target = batch["target"].to(
            DEVICE
        )

        optimizer.zero_grad()

        output = model(
            person,
            cloth,
            pose
        )

        loss = criterion(
            output,
            target
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = (
        epoch_loss
        / len(loader)
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.6f}"
    )

    # save checkpoint

    if (epoch + 1) % 10 == 0:

        torch.save(
            model.state_dict(),
            f"checkpoint_epoch_{epoch+1}.pth"
        )

print()

print("Training Complete")
