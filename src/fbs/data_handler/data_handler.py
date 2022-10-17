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


liste = [1, 123, 234]


class DataHandler:
    def __init__(self) -> None:
        self.data_dir = "src/fbs/data/"
        self.transforms = None

    def prepare_dataloader(self, batch_size):
        train, test = self.prepare_dataset()
        train_size, test_size = len(train), len(test)
        return (
            DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=batch_size),
            train_size,
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
