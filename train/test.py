import torch
import torch.nn as nn

from torchtext import data
from train.train import test
from train.dataprepare import readdata
from train.model import RNN, BaseLine
from torch import optim
from train.configs import BATCH_SIZE, EMBEDDING_SIZE, MAX_VOCAB_SIZE


# from train import test
# from dataprepare import readdata
# from model import BaseLine, RNN
# from torch import optim
# from configs import BATCH_SIZE, EMBEDDING_SIZE, MAX_VOCAB_SIZE

import spacy
nlp = spacy.load('en')

def deploy_model(sentence="This film is terrible",
                 packed=False,
                 pretrain=False,
                 max_vocab_size=MAX_VOCAB_SIZE,
                 embedding_size=EMBEDDING_SIZE):
    print("Input query: '{}'".format(sentence))
    train_data, valid_data, test_data, text, label = readdata(packed=packed,
                                                              pretrain=pretrain,
                                                              max_vocab_size=max_vocab_size)

    model = BaseLine(len(text.vocab), embedding_size, embedding_size, 1)
    model.load_state_dict(torch.load('./models/tut1-model.pt'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [text.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    # prediction = torch.sigmoid(model(tensor, length_tensor))
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def main():
    deploy_model()    

if __name__ == "__main__":
    main()