# Start with a Linux micro-container to keep the image tiny
FROM continuumio/miniconda3

MAINTAINER YL
RUN conda install -y pytorch torchvision -c pytorch

WORKDIR /tensorflow-mnist
ADD requirements.txt  requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en

ADD ./train/  /tensorflow-mnist/train
RUN mkdir /tensorflow-mnist/models
RUN mkdir /tensorflow-mnist/logs
RUN touch server.log

ENV RESULT_DIR='/tensorflow-mnist'

CMD ["python train/train.py -stop True -topology BASELINE > server.log 2>&1"]
