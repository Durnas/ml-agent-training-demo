"""Minimal MNIST training demo for ml-agent's create_training_job.

Purpose: prove the GPU-in-Kubernetes pipeline works end-to-end (image pull,
GPU scheduling, dependency install, live log streaming) — not to produce a
good model. Keep it short: MNIST, small CNN, 2 epochs.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 2
BATCH_SIZE = 64
LR = 1e-3


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root="/tmp/data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"[train] epoch {epoch}/{EPOCHS} batch {batch_idx}/{len(train_loader)} "
                      f"loss {loss.item():.4f}", flush=True)

        print(f"[train] epoch {epoch} done, avg loss {running_loss / len(train_loader):.4f}", flush=True)

    elapsed = time.time() - start
    print(f"[train] finished in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
