import torch
from torch import nn
from tqdm import tqdm

loss_function = nn.CrossEntropyLoss()


def validate(net, val, device):
    net.eval()
    val_loss = 0
    # iterate over the test set
    for data in tqdm(val, total=len(val), desc="Validation"):
        # get inputs; data is list [inputs, labels]
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)
        masks = torch.squeeze(masks, 1).type(torch.long)

        with torch.no_grad():
            pred = net(inputs)
            loss = loss_funtion(pred, masks)
            val_loss += loss.item()

    net.train()
    return val_loss
