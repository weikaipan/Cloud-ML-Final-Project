# Start with a Linux micro-container to keep the image tiny
FROM continuumio/miniconda3

MAINTAINER YL
RUN conda install -y pytorch torchvision -c pytorch

WORKDIR /tensorflow-mnist
ADD requirements.txt  requirements.txt
RUN pip install -r requirements.txt

ADD ./train/  /tensorflow-mnist/train
RUN python -m spacy download en
RUN mkdir /tensorflow-mnist/models

ENV RESULT_DIR='/tensorflow-mnist'

CMD ["python", "train/train.py"]
