import argparse
import random
import torch
import torch.nn as nn
from ..models.CNN_tp1 import CNN
from ..utils.data_loder import train_set,test_data
from ..train.train import train
from ..evaluate.test import test
import wandb

wandb.login(key='731be6e0f444c6f77a3bb899d5cd29f3c7959a49')

def fix_randomness(SEED):
    random.seed(SEED)
    #np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args):
    model = CNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader=train_set(args.batch_size)
    test_loader=test_data(args.batch_size)
     # Apprentissage
    train_acc, test_acc = [], []
    train_loss, test_loss = [], []

    for epoch in range(1, 11):
        train_loss_cur, train_acc_cur = train(model, train_loader, optimizer, criterion, epoch)
        test_loss_cur, test_acc_cur = test(model, test_loader, criterion)
        train_acc.append(train_acc_cur)
        test_acc.append(test_acc_cur)
        train_loss.append(train_loss_cur)
        test_loss.append(test_loss_cur)


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument('-bch', "--batch_size", type=int, default=64,help="size of batch with defaul value 64")
     parser.add_argument('-lr', "--learning_rate", type=float, default=0.001)


     args = parser.parse_args()
     print("=" * 50)
     for arg in vars(args):
        print(arg, '=',getattr(args, arg))
     print("=" * 50)
     run(args)

