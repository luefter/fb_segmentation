import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from torch import nn

from fbs.data_handler import DataHandler
from fbs.unet.unet_model import UNet

EPOCH_COUNT = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.001
VERBOSE = True
VERBOSE_BATCH_COUNT = 40


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Selceted {device} as training device")
    model = UNet(3, 3).to(device).train()
    data_handler = DataHandler()
    trainloader, testloader, _, _ = data_handler.prepare_dataloader(
        batch_size=BATCH_SIZE
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH_COUNT):  # loop over the dataset multiple times

        running_train_loss = 0.0
        running_test_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)

            masks = torch.squeeze((masks - 1), 1).type(torch.long)

            loss = loss_function(outputs, masks)

            loss.backward()
            optimizer.step()

            if VERBOSE:
                running_train_loss += loss.item()
                if (
                    i % VERBOSE_BATCH_COUNT == VERBOSE_BATCH_COUNT - 1
                ):  # print every verbose_batch_count mini-batches
                    print(
                        f"[{epoch + 1}, {i + 1:5d}] loss: {running_train_loss / (VERBOSE_BATCH_COUNT):.3f}"
                    )
                    running_train_loss = 0.0

        del outputs

        for i, data in enumerate(testloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, masks = data
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)

            masks = torch.squeeze((masks - 1), 1).type(torch.long)

            loss = loss_function(outputs, masks)

            running_test_loss += loss.item()

        print(
            f"epoch: {epoch + 1} --- test loss: {running_test_loss / len(testloader):.3f}"
        )

    torch.save(model.state_dict(), f"src/fbs/data/model_weights/model_{EPOCH_COUNT}")


if __name__ == "__main__":
    main()
