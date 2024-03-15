import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, root, split):
        self.data = np.load(f"{root}/mnist.npy").astype("float32")
        # normalize
        self.data = self.data / 255.0
        self.labels = np.load(f"{root}/labels.npy")
        self.labels = self.labels.astype(int)

        # use the first 50000 images for training
        if split == "train":
            self.data = self.data[:50000]
            self.labels = self.labels[:50000]
        elif split == "valid":
            self.data = self.data[55000:]
            self.labels = self.labels[55000:]
        elif split == "test":
            self.data = self.data[50000:55000]
            self.labels = self.labels[50000:55000]
        else:
            raise NotImplementedError

        self.data = self.data.reshape(-1, 1, 28, 28)

        self.dataset = [
            {"image": img, "label": lb} for img, lb in zip(self.data, self.labels)
        ]

    def __getitem__(self, index):
        img = self.dataset[index]["image"]
        lb = self.dataset[index]["label"]
        img = torch.tensor(img)
        lb = torch.tensor(lb)
        return {"image": img, "label": lb}

    def __len__(self):
        return len(self.dataset)


LPNDataset = MNISTDataset
