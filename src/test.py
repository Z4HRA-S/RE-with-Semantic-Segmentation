from src.data_loader import DocRED
from torch.utils.data import DataLoader
from src.model import Model
import torch
from src.capsnet import CapsNet
from torchsummary import summary
import gc
import numpy as np


def test_model():
    training_data = DocRED("dataset/dev.json")
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    model = Model()
    for data in train_dataloader:
        x = data
        break
    del train_dataloader
    del training_data
    gc.collect()
    y=model(x)



def test_caps_net():
    model = CapsNet()
    output, reconstructions, masked = model(torch.rand(5, 3, 42, 42))
    print(output.size(), reconstructions.size(), masked.size())
    print(masked[0].argmax())
    length = torch.Tensor([[torch.norm(vector) for vector in sample] for sample in output])
    print(length.size())
    i = length[0].argmax()
    print(i)
    print(length[0][int(i)])


def summary_caps_net():
    model = CapsNet()
    summary(model, input_size=(3, 42, 42))

test_model()
