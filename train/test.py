import torch
import torch.nn as nn

from torchtext import data
from train import test
from dataprepare import readdata
from model import RNN
from torch import optim
from configs import BATCH_SIZE, EMBEDDING_SIZE

def main():
    train_data, valid_data, test_data, text, label = readdata()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data prepare.
    _, _, test_iterator = data.BucketIterator.splits((train_data,
                                                      valid_data,
                                                      test_data), 
                                                      batch_size=BATCH_SIZE,
                                                      device=device)

    model = RNN(len(text.vocab), EMBEDDING_SIZE, 256, 1)
    model = model.to(device)
    # Criterion
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    test_loss, test_acc = test(model, test_iterator, criterion)

if __name__ == "__main__":
    main()