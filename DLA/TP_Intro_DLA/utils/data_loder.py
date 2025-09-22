from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# chargement dataset
def train_set(btch):
    train_data = datasets.MNIST(
    root = '../data',
    train = True,                         
    transform = transforms.ToTensor(), 
    download = True,)
    return DataLoader(train_data, batch_size=btch, shuffle=True)


def test_data(btch):
    test_data = datasets.MNIST(
        root = '../data', 
        train = False, 
        transform = transforms.ToTensor()
    )
    return  DataLoader(test_data, batch_size=btch, shuffle=False)


# Combien d'images dans chaque ensemble ?