import torch
import torch.nn as nn

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    current_loss = []
    nb_ok = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        nb_ok += (output.argmax(dim=1) == target).float().sum()
        loss = criterion(output, target)
        current_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    current_loss = sum(current_loss)/len(current_loss)
    print(f"Epoch {epoch} - loss: {current_loss:.2f}")
    acc_train = nb_ok/len(train_loader.dataset)
    print(f"Accuracy: {acc_train:.2f}")
    return current_loss, acc_train