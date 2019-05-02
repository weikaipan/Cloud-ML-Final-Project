# Start with a Linux micro-container to keep the image tiny
FROM continuumio/miniconda3

MAINTAINER YL
RUN conda install -y pytorch torchvision -c pytorch

WORKDIR /tensorflow-mnist
ADD ./train/  /tensorflow-mnist
ADD requirement.txt  requirement.txt

RUN pip install -r requirement.txt
RUN python -m spacy download en

ENV RESULT_DIR='/tensorflow-mnist'


CMD ["python", "train.py"]
