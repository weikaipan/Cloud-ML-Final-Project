"""This is the module for preparing data."""
import torch
import random
from torchtext import data
from torchtext import datasets
from configs import MAX_VOCAB_SIZE

def readdata():
    print("Reading Data")
    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    text = data.Field(tokenize='spacy')
    label = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(text, label)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))
    
    text.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    label.build_vocab(train_data)
    
    print(text.vocab.itos[:10])
    print(label.vocab.stoi)
    print("==== Chcek an Example ====")
    print(vars(train_data.examples[0]))
    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    return train_data, valid_data, test_data

if __name__ == '__main__':
    # for testing purpose.
    train_data, test_data = readdata()
