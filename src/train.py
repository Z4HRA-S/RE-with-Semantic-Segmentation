from src.data_loader import DataLoader, DocRED
from src.model import Model
import torch


# todo: see how semantic seg handle loss and labels

def train(loss_fn, optimizer):
    running_loss = 0.
    last_loss = 0.

    training_data = DocRED("dataset/dev.json")
    train_dataloader = DataLoader(training_data, batch_size=5, shuffle=True)

    model = Model()
    for data in train_dataloader:
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(data)

        # Compute the loss and its gradients
        labels = [d["label"] for d in data]
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
