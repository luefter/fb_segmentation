import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    PILToTensor,
    Resize,
    ToTensor,
)


def remove_border(trimap: torch.Tensor):
    return (trimap != 2).astype(torch.uint8)


class DataHandler:
    def __init__(self) -> None:
        self.data_dir = "src/fbs/data/"
        self.transforms = None

    def prepare_dataloader(self, batch_size, val_split=0.1):

        train, test = self.prepare_dataset()
        train_size, test_size = len(train), len(test)
        # split training data into train and validation set
        n_val = int(train_size * val_split)
        n_train = train_size - n_val
        train_set, val_set = torch.utils.data.random_split(
            train, [n_train, n_val], generator=torch.Generator().manual_seed(0)
        )

        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=True),
            DataLoader(test, batch_size=batch_size),
            n_train,
            test_size,
        )

    @staticmethod
    def reduce_classes(img: PIL.Image) -> torch.Tensor:
        """Reduce set of classes from {1,2,3} to {0,1}"""
        return torch.unsqueeze(
            torch.from_numpy((np.array(img) != 2).astype(np.uint8)), 0
        )

    def prepare_dataset(self):
        train = torchvision.datasets.OxfordIIITPet(
            root=self.data_dir,
            target_types="segmentation",
            split="trainval",
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Resize((224, 224), InterpolationMode.BILINEAR),
                ]
            ),
            target_transform=Compose(
                [
                    self.reduce_classes,
                    Resize((224, 224), InterpolationMode.NEAREST),
                ]
            ),
        )

        test = torchvision.datasets.OxfordIIITPet(
            root=self.data_dir,
            target_types="segmentation",
            split="test",
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    Resize((224, 224), InterpolationMode.BILINEAR),
                ]
            ),
            target_transform=Compose(
                [
                    self.reduce_classes,
                    Resize((224, 224), interpolation=InterpolationMode.NEAREST),
                ]
            ),
        )
        return train, test
