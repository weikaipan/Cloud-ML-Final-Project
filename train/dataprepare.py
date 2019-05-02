"""This is the module for preparing data."""
import torch
import random
import time
from torchtext import data
from torchtext import datasets
from configs import MAX_VOCAB_SIZE
from utils import epoch_time

def readdata(pretrain=False):
    print("Reading Data")
    start_time = time.time()
    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # add packed padded sequence.
    text = data.Field(tokenize='spacy', include_lengths=True)
    label = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(text, label)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    if pretrain:
        # use glove, initialize unknown words as random tensor, using gaussian distribution.
        text.build_vocab(train_data, max_size=MAX_VOCAB_SIZE, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
    else:
        text.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    label.build_vocab(train_data)
    
    end_time = time.time()
    read_mins, read_secs = epoch_time(start_time, end_time)

    print(text.vocab.itos[:10])
    print(label.vocab.stoi)
    print("==== Chcek an Example ====")
    print(vars(train_data.examples[0]))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')
    print('Read Time: {}m {}s'.format(read_mins, read_secs))

    return train_data, valid_data, test_data, text, label

if __name__ == '__main__':
    # for testing purpose.
    train_data, test_data = readdata()
