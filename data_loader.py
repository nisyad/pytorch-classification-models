import os
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class Cifar10Dataset(Dataset):
    def __init__(self, files, label_mapping, transform=None):

        self.files = files
        self.data_size = len(self.files)
        self.label_mapping = label_mapping
        # self.files = random.sample(files, self.data_size)
        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        image = np.array(image)

        label_name = image_address[:-4].split("_")[-1]
        label = self.label_mapping[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def fetch(data_dir, train=True, batch_size=128):

    # Create label mapping
    with open(data_dir + "/labels.txt") as label_file:
        labels = label_file.read().split()
        label_mapping = dict(zip(labels, list(range(len(labels)))))

    if train:

        data_dir = data_dir + "/train"

    else:
        data_dir = data_dir + "/test"

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, x) for x in files]

    dataset = Cifar10Dataset(files=files,
                             label_mapping=label_mapping,
                             transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=1)

    return dataloader
