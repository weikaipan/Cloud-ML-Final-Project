# Start with a Linux micro-container to keep the image tiny
FROM continuumio/miniconda3

MAINTAINER YL
RUN conda install -y pytorch torchvision -c pytorch

WORKDIR /cloudfinal
ADD requirements.txt  requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en

ADD train train
ADD static static
ADD app.py app.py
ADD settings.py settings.py
RUN mkdir /cloudfinal/models

ENV RESULT_DIR='/cloudfinal'

EXPOSE 8000
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:8000", "app:app" ,"--timeout=90"]