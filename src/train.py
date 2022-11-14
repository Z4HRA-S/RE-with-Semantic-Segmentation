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
            loss, current = loss.item(), (batch + 1) * len(data)
            print(f"loss: {loss:>7f} f1: {f1:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)

    with torch.no_grad():
        for data in dataloader:
            pred = model(data)
            loss, f1 = loss_fn(pred, data["label"])

    loss /= num_batches
    print(f"Test Error: \n F1: {f1 :>7f}, Avg loss: {loss:>8f} \n")

############################################
device = torch.device("cuda:0")

epochs = 30
batch_size = 4
learning_rate = 4e-4
adam_epsilon = 1e-6
bert_lr = 3e-5

loss_fn = balanced_loss()
loss_fn.to(device)

training_data = DocRED("dataset/train_annotated.json")
train_size = int(0.8 * len(training_data))
test_size = len(training_data) - train_size
train_data, test_data = torch.utils.data.random_split(training_data, [train_size, test_size])


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

model = Model()
model.to(device)

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
