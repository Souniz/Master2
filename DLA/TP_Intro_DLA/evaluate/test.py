import torch
import torch.nn as nn

def test(model, test_loader, criterion):
    model.eval()
    test_loss = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            cur_loss = criterion(output, target)
            test_loss.append(cur_loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss = sum(test_loss)/len(test_loader)
    acc_loss = correct/len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.2f}, Accuracy: {acc_loss:.2f}")
    return test_loss, acc_loss