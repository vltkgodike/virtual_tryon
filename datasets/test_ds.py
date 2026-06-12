from datasets.viton_dataset import VITONDataset

dataset = VITONDataset(
    root_dir=r"C:\Users\valkontek 010\Downloads\VTON-HD"
)

print("Dataset Size:")
print(len(dataset))

sample = dataset[0]

print()

print("Person Shape:")
print(sample["person"].shape)

print()

print("Cloth Shape:")
print(sample["cloth"].shape)

print()

print("Target Shape:")
print(sample["target"].shape)

print(
    sample["pose"].shape
)