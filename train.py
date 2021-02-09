import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import ResNet
import data_loader

# Hyperparameters
EPOCHS = 1
BATCH_SIZE = 512
LR = 0.01
WEIGHT_DECAY = 5e-4

DATA_DIR = "data/cifar"

trainloader = data_loader.fetch(data_dir=DATA_DIR,
                                train=True,
                                batch_size=BATCH_SIZE)
testloader = data_loader.fetch(data_dir=DATA_DIR,
                               train=False,
                               batch_size=BATCH_SIZE)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = ResNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),
                      lr=LR,
                      momentum=0.9,
                      weight_decay=WEIGHT_DECAY)


def train():
    # Training Loop
    for epoch in range(EPOCHS):
        losses = []

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if batch_idx % 200 == 0:
                print(
                    f"Epoch: [{epoch} / {EPOCHS}]  Batch Index: {batch_idx}  Loss: {np.mean(losses)}"
                )

        # Evaluation
        net.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)

                _, predicted = torch.max(outputs.detach(), dim=1)
                total += targets.shape[0]
                correct += predicted.eq(targets).cpu().sum()

                print(
                    f"Epoch: [{epoch} / {EPOCHS}]  Test Accuracy: {100*correct/total}"
                )


if __name__ == '__main__':
    train()
