import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from loguru import logger
from torch import nn
from tqdm import tqdm

from fbs.data_handler import DataHandler
from fbs.unet.unet_model import UNet
from fbs.validate import validate

# Hyperparameter
EPOCH_COUNT = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.001
VERBOSE = True
VERBOSE_BATCH_COUNT = 40


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Selceted {device} as training device")
    model = UNet(3, 2).to(device).train()
    data_handler = DataHandler()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # load train, val and test data
    train, val, test, n_train, _ = data_handler.prepare_dataloader(
        batch_size=BATCH_SIZE
    )
    train_loss_records = []
    val_loss_records = []
    weights = deque(maxlen=3)

    # begin training
    for epoch in range(1, EPOCH_COUNT + 1):
        model.train()
        epoch_loss = 0
        for data in tqdm(train, total=len(train), desc=f"Epoch {epoch}/{EPOCH_COUNT}"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)

            masks = torch.squeeze(masks, 1).type(torch.long)

            loss = loss_function(outputs, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # validation round
        val_loss = validate(model, val, device)
        train_loss_records.append(epoch_loss)
        val_loss_records.append(val_loss)
        weights.append(model.state_dict())
        print(
            f"Epoch:{epoch}, training loss: {epoch_loss/len(train)}, validation loss: {val_loss/len(val)}"
        )
        if epoch > 3 and val_loss_records[-1] > val_loss_records[-3]:
            torch.save(weights[0], f"src/fbs/data/model_weights/model_{EPOCH_COUNT}")
            break

    #     # Test loop
    #     for i, data in enumerate(testloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, masks = data
    #         inputs, masks = inputs.to(device), masks.to(device)
    #         outputs = model(inputs)
    #
    #         masks = torch.squeeze(masks, 1).type(torch.long)
    #
    #         loss = loss_function(outputs, masks)
    #
    #         running_test_loss += loss.item()
    #
    #     print(
    #         f"epoch: {epoch + 1} --- test loss: {running_test_loss / len(testloader):.3f}"
    #     )
    #
    # torch.save(model.state_dict(), f"src/fbs/data/model_weights/model_{EPOCH_COUNT}")


if __name__ == "__main__":
    main()
