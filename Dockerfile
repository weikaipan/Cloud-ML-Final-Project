# Start with a Linux micro-container to keep the image tiny
FROM frolvlad/alpine-miniconda2

# Document who is responsible for this image
MAINTAINER Yonghao Liu
RUN conda install -y pytorch torchvision -c pytorch

ADD ./train/  /tensorflow-mnist
ADD requirements.txt  requirements.txt

RUN pip install -r requirements.txt

WORKDIR /tensorflow-mnist
ENV RESULT_DIR='./tensorflow-mnist'



CMD ["python", "train.py"]
