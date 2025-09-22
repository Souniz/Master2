# Definition du modele
import torch
import torch.nn as nn

"""
    ##  Création du Réseau CNN

   Implémentez une classe `CNN` qui hérite de `nn.Module`. Cette classe doit comporter les couches suivantes :
   - Une première couche de convolution `conv1` qui prend en entrée des images en niveaux de gris et génère 16 canaux de sortie. Utilisez un noyau de convolution de taille 3x3 avec un padding de 1.
   - Une couche d'activation ReLU.
   - Une couche de pooling maximal `max_pool_2D` avec une fenêtre de 2x2 et un stride de 2.
   - Une deuxième couche de convolution `conv2` qui prend en entrée les 16 canaux de sortie de la première couche de convolution et génère 32 canaux de sortie, avec les mêmes paramètres de noyau et de padding.
   - Une autre couche d'activation ReLU.
   - Une autre couche de pooling maximal de même type que le premier.
   - Une couche entièrement connectée `fc1` qui prend en entrée les sorties aplaties des couches précédentes et a une dimension 128 en sortie.
   - Une autre couche entièrement connectée `fc2` avec autant de dimensions en sortie que de classes dans MNIST.
    """
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x