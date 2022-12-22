from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
import logging
import numpy as np
from mmd import cal_mmd

# from vit_pytorch.efficient import ViT
from mod_vit import ViT

# set seed
seed = 2021

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = torch.device('cuda:0')

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

train_data = datasets.ImageFolder('train_path', transform=train_transforms)
test_data = datasets.ImageFolder('test_path', transform=test_transforms)

batch_size = 128
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)



model = ViT(
    channels = 3,
    image_size = 224,
    patch_size = 32,
    num_classes = 3,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(device)

logging.basicConfig(level=logging.INFO,
                    filename = "log.log",
                    format='[%(asctime)s] - %(message)s')

epochs = 100
lr = 1e-3
gamma = 0.7

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

accs = []
best_acc = 0

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        u,s,v = torch.linalg.svd(data)
        data_ = torch.matmul(torch.matmul(u[:, :, :, :10], torch.diag_embed(s)[:, :, :, :10]), v.transpose(-2, -1)[:, :, :, :10])
        data = data.to(device)
        data_ = data_.to(device)
        label = label.to(device)

        output, mid = model(data)
        output_, mid_ = model(data_)
        loss = criterion(output, label) + 0.001* cal_mmd(mid, mid_)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)

            test_output, _ = model(data)
            test_loss = criterion(test_output, label)

            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)

    if epoch_test_accuracy > best_acc:
        print(epoch_test_accuracy)
        print(best_acc)
        best_acc = epoch_test_accuracy
        torch.save(obj=model.state_dict(), f='model.pth')

    logging.info(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - test_loss : {epoch_test_loss:.4f} - test_acc: {epoch_test_accuracy:.4f}\n"
    )

    accs.append(epoch_test_accuracy)

np.save('loss.npy', accs)