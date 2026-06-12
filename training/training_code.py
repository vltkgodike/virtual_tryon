import os
import sys
import torch
import torch.nn as nn
from torchvision.utils import save_image

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

EPOCHS = 100

BATCH_SIZE = 1

LR = 1e-4

SAVE_EVERY = 10

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


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

preview_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)

preview_batch = next(
    iter(
        preview_loader
    )
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

    # save checkpoint and sample output

    if (epoch + 1) % SAVE_EVERY == 0:

        torch.save(
            model.state_dict(),
            os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )
        )

        model.eval()

        with torch.no_grad():

            preview_output = model(
                preview_batch["person"].to(
                    DEVICE
                ),
                preview_batch["cloth"].to(
                    DEVICE
                ),
                preview_batch["pose"].to(
                    DEVICE
                )
            )

            save_image(
                preview_output.cpu().clamp(
                    0,
                    1
                ),
                os.path.join(
                    OUTPUT_DIR,
                    f"epoch_{epoch+1}.png"
                )
            )

print()

print("Training Complete")
