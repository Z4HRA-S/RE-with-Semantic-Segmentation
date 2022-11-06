from src.data_loader import DocRED
from torch.utils.data import DataLoader
from src.model import Model
from src.loss import balanced_loss
import torch
from transformers.optimization import AdamW


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        y = data["label"]
        pred = model(data)
        loss, f1 = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(data)
            print(f"loss: {loss:>7f} f1: {f1:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for data in dataloader:
            pred = model(data)
            test_loss += loss_fn(pred, data["label"]).item()[0]
            correct += (pred.argmax(1) == data["label"]).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 30
batch_size = 4
learning_rate = 4e-4
adam_epsilon = 1e-6
bert_lr = 3e-5

loss_fn = balanced_loss()
training_data = DocRED("dataset/dev.json")
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

test_data = DocRED("dataset/test.json")
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = Model()

#####
cur_model = model.module if hasattr(model, 'module') else model
extract_layer = ["extractor", "bilinear"]
bert_layer = ['bert_model']
optimizer_grouped_parameters = [
    {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": bert_lr},
    {"params": [p for n, p in cur_model.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
    {"params": [p for n, p in cur_model.named_parameters() if not any(nd in n for nd in extract_layer + bert_layer)]},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

#####

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
