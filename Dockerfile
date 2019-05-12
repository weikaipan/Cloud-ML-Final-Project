# Start with a Linux micro-container to keep the image tiny
FROM continuumio/miniconda3

MAINTAINER YL
RUN conda install -y pytorch torchvision -c pytorch

WORKDIR /cloudfinal
ADD requirements.txt  requirements.txt
RUN pip install -r requirements.txt
RUN python -m spacy download en

COPY train train
COPY static static
COPY app.py app.py
COPY config.py config.py
COPY settings.py settings.py
COPY kubernetes_jobs.py kubernetes_jobs.py
COPY deploy deploy
RUN mkdir /cloudfinal/models
ENV RESULT_DIR='/cloudfinal'

EXPOSE 8000
CMD ["gunicorn", "--workers=2", "--bind=0.0.0.0:8000", "app:app" ,"--timeout=90"]