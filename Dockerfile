# Start with a Linux micro-container to keep the image tiny
FROM alpine:3.7
FROM tensorflow/tensorflow
# Document who is responsible for this image
MAINTAINER Wei-Kai Pan "wkp219@nyu.edu"

WORKDIR /tensorflow-mnist

ENV RESULT_DIR='/tensorflow-mnist'

# Set up a working folder and install the pre-reqs
# WORKDIR /app
COPY /tf-model/convolutional_network.py /tensorflow-mnist/convolutional_network.py
COPY /tf-model/input_data.py /tensorflow-mnist/input_data.py
COPY /t10k-images-idx3-ubyte.gz /tensorflow-mnist/t10k-images-idx3-ubyte.gz
COPY /t10k-labels-idx1-ubyte.gz /tensorflow-mnist/t10k-labels-idx1-ubyte.gz
COPY /train-images-idx3-ubyte.gz /tensorflow-mnist/train-images-idx3-ubyte.gz
COPY /train-labels-idx1-ubyte.gz /tensorflow-mnist/train-labels-idx1-ubyte.gz

CMD ["python", "convolutional_network.py", "--trainImagesFile", "/tensorflow-mnist/train-images-idx3-ubyte.gz", "--trainLabelsFile", "/tensorflow-mnist/train-labels-idx1-ubyte.gz", "--testImagesFile", "/tensorflow-mnist/t10k-images-idx3-ubyte.gz", "--testLabelsFile", "/tensorflow-mnist/t10k-labels-idx1-ubyte.gz", "--learningRate 0.001", " --trainingIters", "20000"]
