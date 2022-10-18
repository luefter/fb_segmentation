import torch
import os
import PIL
from torchvision.transforms import ToPILImage
from fbs.unet.unet_model import UNet
from fbs.data_handler import DataHandler
import matplotlib.pyplot as plt
import argparse
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    PILToTensor,
    Resize,
    ToTensor,
    ToPILImage,
)

IMAGE_PATH = "src/fbs/data/oxford-iiit-pet/images/"
MODEL_PATH = "src/fbs/data/model_weights/model_5"
EXAMPLES_PATH = "src/fbs/data/examples"


def inference(img_name: str = "Abyssinian_1.jpg", save: bool = False) -> PIL.Image:
    # initialize pretrained model in evaluation mode
    unet = UNet(3, 2)
    unet.load_state_dict(torch.load(MODEL_PATH))
    unet.eval()
    # input parser
    input_parser = Compose(
        [
            ToTensor(),
            Resize((224, 224), InterpolationMode.BILINEAR),
        ]
    )

    with PIL.Image.open(os.path.join(IMAGE_PATH, img_name)) as img:
        img = input_parser(img)

    with torch.no_grad():
        mask_pred = unet(torch.unsqueeze(img, 0))

    img = ToPILImage()(img)

    mask_color = PIL.Image.new(mode="RGB", size=img.size, color=(0, 255, 255))
    mask_position = ToPILImage()((torch.argmax(mask_pred, 1) * 100).type(torch.uint8))

    image_with_mask = PIL.Image.composite(mask_color, img, mask_position)

    if save:
        image_with_mask.save(os.path.join(EXAMPLES_PATH, "masked_" + img_name))

    return image_with_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-name", type=str, default="Abyssinian_1.jpg")
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()

    # check if image exists in dataset
    assert args.img_name in os.listdir(
        IMAGE_PATH
    ), f"{args.img_name} is not in the dataset"

    image_with_mask = inference(img_name=args.img_name, save=args.save)
