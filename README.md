# Cloud-ML-Final-Project

## Installation
- Docker Image

```
docker build . -t cloudfinal
docker run cloudfinal python train/train.py -stop True -topology BASELINE
```

## Arguments
```
    -embed EMBEDDING_SIZE,    the hidden size for embedding,, default = 600
    -lr LR,                   initial learning rate, default = 0.01
    -batch BATCH_SIZE,        batch size, default = 2
    -layer LAYER_DEPTH,       the depth of recurrent units, default = 1
    -topology                 default is RNN. Support ["BASELINE", "RNN", "GRU"]. "BASELINE" is a recurrent layer

    -gradclip GRAD_CLIP,      gradient clipping, default = 2
    -pretrain True | False,   use pre-train word embedding (GloVe)
    -epoch EPOCH_TIME,        maximum epoch time for training
    -optim Adam | Adagrad     default is SGD
    -stop True | False        for testing purpose, True for early stop and false for full training
    -packed True | False      using packed padded sequence if true, otherwise false
    
    -getloss GET_LOSS,        print out average loss every `GET_LOSS` iterations.
    -epochsave SAVE_MODEL,    save the model every `SAVE_MODEL` epochs.
    -outputfile OUTPUT_FILE,  root folder for saving trained models
```
