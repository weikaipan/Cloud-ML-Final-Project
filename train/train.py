"""This is core training part, containing different models.
Modify:
https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb
"""
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch import optim
from torchtext import data
from train.configs import BATCH_SIZE, EMBEDDING_SIZE, LR, LAYER_DEPTH, CNN_N_FILTERS
from train.configs import SAVE_MODEL, OUTPUT_FILE, GRAD_CLIP, EPOCH
from train.configs import GET_LOSS, MAX_TRAIN_NUM, MAX_VOCAB_SIZE, PRETRAIN, OPTIM
from train.dataprepare import readdata
from train.model import RNN, BaseLine, CNN
from pprint import pprint
from train.utils import epoch_time

# from app import celery

# from configs import BATCH_SIZE, EMBEDDING_SIZE, LR, LAYER_DEPTH, CNN_N_FILTERS
# from configs import SAVE_MODEL, OUTPUT_FILE, GRAD_CLIP, EPOCH
# from configs import GET_LOSS, MAX_TRAIN_NUM, MAX_VOCAB_SIZE, PRETRAIN, OPTIM
# from dataprepare import readdata
# from model import RNN, BaseLine, CNN
# from pprint import pprint
# from utils import epoch_time
from celery import Celery
from app import app
# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def pick_optimizer(optim_option, parameters, learning_rate):
    if optim_option == 'Adam':
        return optim.Adam(parameters, lr=learning_rate)
    elif optim_option == 'Adagrad':
        return optim.Adagrad(parameters,
                             lr=learning_rate,
                             lr_decay=0,
                             weight_decay=0)
    else:
        return optim.SGD(parameters, lr=learning_rate)

def test(model, test_iterator, criterion, stop=False, packed=False):
    model.load_state_dict(torch.load(OUTPUT_FILE + '/tut1-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion, stop=stop, packed=packed)
    print('Test loss = {}, Test Acc = {}'.format(test_loss, test_acc * 100))
    return test_loss, test_acc

def evaluate(model, iterator, criterion, stop=False, packed=False):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            if packed:
                text, text_len = batch.text
                predictions = model(text, text_len).squeeze(1)
            else:
                predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if stop:
                break

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train(model, iterator, optimizer, criterion, stop=False, packed=False):
    """."""
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for batch in iterator:

        optimizer.zero_grad()

        # use pack padded sequence.
        if packed:
            text, text_len = batch.text
            predictions = model(text, text_len).squeeze(1)
        else:
            predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if stop:
            break

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

@celery.task(bind=True)
def main(self,
         embedding_size=EMBEDDING_SIZE,
         n_filters=CNN_N_FILTERS,
         learning_rate=LR,
         batch_size=BATCH_SIZE,
         get_loss=GET_LOSS,
         grad_clip=GRAD_CLIP,
         epoch=EPOCH,
         layer_depth=LAYER_DEPTH,
         save_model=SAVE_MODEL,
         output_file=OUTPUT_FILE,
         pretrain=PRETRAIN,
         optim_option=OPTIM,
         topology='BASELINE',
         stop=False,
         packed=False,
         max_vocab_size=MAX_VOCAB_SIZE):
    """Main train driver."""

    self.update_state(state='READING',
                      meta={ 'INFO': 'READING DATA' })

    train_data, valid_data, test_data, text, label = readdata(packed=packed,
                                                              pretrain=pretrain,
                                                              max_vocab_size=max_vocab_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data prepare.
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_data,
                                                                                valid_data,
                                                                                test_data),
                                                                                sort_within_batch=True,
                                                                                batch_size=BATCH_SIZE,
                                                                                device=device)
    print("Start Training")
    if topology == 'BASELINE':
        model = BaseLine(len(text.vocab), embedding_size, embedding_size, 1)
    elif topology == 'CNN':
        model = CNN(len(text.vocab), [3,4,5], topology=topology)
    else:
        model = RNN(len(text.vocab), embedding_size, embedding_size, 1, topology=topology)

    model = model.to(device)
    # Criterion
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    optimizer = pick_optimizer(optim_option, model.parameters(), learning_rate)

    # Training
    best_valid_loss = float('inf')

    self.update_state(state='TRAIN',
                      meta={ 'INFO': 'START TRAINING' })
    training_result = { 'Epoch': 'Pending', 'Train': 'Pending', 'Val': 'Pending'}
    for epoch in range(epoch):
        
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, stop=stop, packed=packed)

        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, stop=stop, packed=packed)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), OUTPUT_FILE + '/tut1-model.pt')

        print('Epoch: {} | Epoch Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs))
        print('Train Loss: {} | Train Acc: {}%'.format(train_loss, train_acc * 100))
        print('Val. Loss: {} |  Val. Acc: {}%'.format(valid_loss, valid_acc * 100))
        training_result = { 'Epoch': 'Epoch: {} | Epoch Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs),
                            'Train': 'Train Loss: {} | Train Acc: {}%'.format(train_loss, train_acc * 100),
                            'Val': 'Val. Loss: {} |  Val. Acc: {}%'.format(valid_loss, valid_acc * 100)}
        
        self.update_state(state='PROGRESS',
                          meta=training_result)

        if stop:
            return training_result

    test_loss, test_acc = test(model, test_iterator, criterion, stop=stop, packed=packed)
    print("Finished Training")
    training_result = { 'Epoch': 'Epoch: {} | Epoch Time: {}m {}s'.format(epoch, epoch_mins, epoch_secs),
                        'Test': 'Test Acc: {}%'.format(train_acc * 100)}
    self.update_state(state='END',
                      meta=training_result)
    return training_result

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
                    default=GET_LOSS)

    ap.add_argument("-topology",
                    "--topology",
                    default='BASELINE')

    ap.add_argument("-epochsave",
                    "--save_model",
                    type=int,
                    default=SAVE_MODEL)

    ap.add_argument("-pretrain",
                    "--pretrain",
                    type=bool,
                    default=PRETRAIN)

    ap.add_argument("-outputfile",
                    "--output_file",
                    default=OUTPUT_FILE)

    ap.add_argument("-gradclip",
                    "--grad_clip",
                    type=int,
                    default=GRAD_CLIP)

    ap.add_argument("-epoch",
                    "--epoch",
                    type=int,
                    default=EPOCH)

    ap.add_argument("-optim",
                    "--optim_option",
                    default=OPTIM)

    ap.add_argument("-stop",
                    "--stop",
                    type=bool,
                    default=False)

    ap.add_argument("-packed",
                    "--packed",
                    type=bool,
                    default=False)

    ap.add_argument("-maxvocabsize",
                    "--max_vocab_size",
                    type=int,
                    default=25000)

    return ap.parse_args()

def print_settings(args):
    print("---------------")
    print("Parameter Settings:")
    for arg in vars(args):
        print("{} = {}".format(arg, vars(args)[arg]))
    print("---------------")

if __name__ == '__main__':
    args = parse_argument()
    print_settings(args)
    result = main(**vars(args))
