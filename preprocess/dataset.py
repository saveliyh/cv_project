from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch
from transforms import train_transform
from os import path


class TUDataset(Dataset):
    def __init__(self, annotations: list[dict], transform, path_to_files: str):
        self.data = annotations
        self.transform = transform
        self.path = path_to_files

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image = read_image(path.join(self.path, data["raw_file"])).double()

        image = self.transform(image)
        return (image, data["lanes"], data["h_samples"])


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    with open("data/TUSimple/train_set/label_data_0313.json") as f:
        annotations = []
        for line in f.readlines():
            annotations.append(json.loads(line))

    dataset = TUDataset(annotations, train_transform, "data/TUSimple/train_set")

    print(len(dataset))

    plt.imshow(dataset[0][0][0], cmap="gray")
    plt.show()
