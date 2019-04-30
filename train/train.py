"""This is core training part, containing different models."""
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch import optim
from torchtext import data
from configs import BATCH_SIZE, EMBEDDING_SIZE, LR, LAYER_DEPTH
from configs import SAVE_MODEL, OUTPUT_FILE, GRAD_CLIP, EPOCH, GET_LOSS, MAX_TRAIN_NUM
from dataprepare import readdata

from pprint import pprint

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    """."""
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def main(args):
    """Main train driver."""
    train_data, valid_data, test_data = readdata()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args['max_train_nums'] is not None:
        mx_train = args['max_train_nums']
        train_data = train_data[:mx_train]
    del(args['max_train_nums'])

    # Data prepare.
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data,
                                                                                valid_data,
                                                                                test_data),
                                                                                batch_size=BATCH_SIZE,
                                                                                device=device)
    print("Start Training")

    for batch in train_iterator:
        pprint(batch.text)
        pprint(batch.text.shape)
        quit()
    # Optimizer
    # optimizer = optim.SGD(model.parameters(), lr=)
    # optimizer = optim.Adagrad(model.parameters(),
    #                           lr=learning_rate, lr_decay=0, weight_decay=0)

    # optimizer = optim.Adam(model.parameters(),
    #                        lr=learning_rate)

    # Criterion
    # criterion = nn.BCEWithLogitsLoss()

    # model = model.to(device)
    # criterion = criterion.to(device)

    # Train
    # train(model, train_iterator, optimizer, criterion)

def parse_argument():
    """Hyperparmeter tuning."""

    ap = argparse.ArgumentParser()
    ap.add_argument("-embed",
                    "--embedding_size",
                    type=int,
                    default=EMBEDDING_SIZE)

    ap.add_argument("-lr",
                    "--learning_rate",
                    type=float,
                    default=LR)

    ap.add_argument("-batch",
                    "--batch_size",
                    type=int,
                    default=BATCH_SIZE)

    ap.add_argument("-layer",
                    "--layer_depth",
                    type=int,
                    default=LAYER_DEPTH)

    ap.add_argument("-getloss",
                    "--get_loss",
                    type=int,
                    default=GET_LOSS)

    ap.add_argument("-maxtrain",
                    "--max_train_nums",
                    type=int,
                    default=MAX_TRAIN_NUM)

    ap.add_argument("-epochsave",
                    "--save_model",
                    type=int,
                    default=SAVE_MODEL)

    ap.add_argument("-outputfile",
                    "--output_file",
                    default=OUTPUT_FILE)

    ap.add_argument("-gradclip",
                    "--grad_clip",
                    type=int,
                    default=GRAD_CLIP)

    ap.add_argument("-epoch",
                    "--epoch_time",
                    type=int,
                    default=EPOCH)

    return ap.parse_args()

if __name__ == '__main__':
    args = parse_argument()
    main(args)
