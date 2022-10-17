import torch
from torchvision.transforms import ToPILImage
from fbs.unet.unet_model import UNet
from fbs.data_handler import DataHandler
import matplotlib.pyplot as plt

PATH = "src/fbs/data/model_weights/model_5"

# initialize pretrained model
unet = UNet(3, 2)
unet.load_state_dict(torch.load(PATH))

# set model to eval model
unet.eval()

# select image
dh = DataHandler()
_, dataset_test = dh.prepare_dataset()

img, mask_true = dataset_test[2]


with torch.no_grad():
    mask_pred = unet(torch.unsqueeze(img, 0))


img = ToPILImage()(img)
mask_true_img = ToPILImage()(mask_true * 200)
mask_pre_img = ToPILImage()((torch.argmax(mask_pred, 1) * 200).type(torch.uint8))
