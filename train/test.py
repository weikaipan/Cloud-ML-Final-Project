import torch
import torch.nn as nn

from torchtext import data
from train.dataprepare import readdata
from train.model import RNN, BaseLine, CNN
from torch import optim

from train.configs import BATCH_SIZE, EMBEDDING_SIZE, LR, LAYER_DEPTH, CNN_N_FILTERS, HIDDEN_DIM, OUTPUT_DIM, DROPOUT
from train.configs import SAVE_MODEL, OUTPUT_FILE, GRAD_CLIP, EPOCH, MAX_VOCAB_SIZE, HIDDEN_DIM
from train.configs import GET_LOSS, MAX_TRAIN_NUM, MAX_VOCAB_SIZE, PRETRAIN, OPTIM, BIDIRECTIONAL

# from train import test
# from dataprepare import readdata
# from model import BaseLine, RNN
# from torch import optim
# from configs import BATCH_SIZE, EMBEDDING_SIZE, MAX_VOCAB_SIZE

import spacy
nlp = spacy.load('en')

def deploy_model(sentence="This film is terrible",
                 topology='example',
                 packed=False,
                 pretrain=True,
                 max_vocab_size=MAX_VOCAB_SIZE,
                 embedding_size=EMBEDDING_SIZE):
    print("Input query: '{}'".format(sentence))
    if topology == 'lstm':
        packed = True
    else:
        packed = False

    train_data, valid_data, test_data, text, label = readdata(packed=packed,
                                                              pretrain=pretrain,
                                                              max_vocab_size=max_vocab_size)

    if topology == 'cnn':
        model = CNN(len(text.vocab), [3,4,5], topology=topology)
        model_path = './models/CNN_pretrain_81p6.pt'
    elif topology == 'lstm':
        model = RNN(len(text.vocab),
                    HIDDEN_DIM,
                    embedding_dim=EMBEDDING_SIZE,
                    n_layers=LAYER_DEPTH,
                    bidirectional=BIDIRECTIONAL,
                    dropout=DROPOUT,
                    # pad_idx=PAD_IDX,
                    output_dim=OUTPUT_DIM,
                    topology='RNN')
        model_path = './models/RNN_lstm_nopretrain_73p0.pt'
    elif topology == 'rnn':
        model = BaseLine(len(text.vocab), HIDDEN_DIM, embedding_size, 1)
        model_path = './models/RNN_nopretrain_64p9.pt'
    else:
        model = BaseLine(len(text.vocab), embedding_size, embedding_size, 1)
        model_path = './models/tut1-model.pt'

    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [text.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    # prediction = torch.sigmoid(model(tensor, length_tensor))
    if packed:
        prediction = torch.sigmoid(model(tensor, length))
    else:
        prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def main():
    deploy_model()    

if __name__ == "__main__":
    main()