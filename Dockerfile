#FROM pytorch/pytorch
FROM tensorflow/tensorflow:nightly-gpu-py3
#FROM tensorflow/tensorflow:latest
#FROM nvidia/cuda:8.0-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
        build-essential \
        cmake \
        ffmpeg \
        git \
        htop \
        libcupti-dev \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        tree \
        unzip \
        vim \
        wget

RUN pip3 install --upgrade pip
RUN pip3 install \
        future \
        ipython \
        matplotlib \
        nltk \
        nose \
        numpy \
        pandas \
        requests \
        scikit-image \
        scipy \
        sk-video \
        sortedcontainers \
        tornado \
        wheel

# install PyTorch
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

# install TensorFlow
#RUN pip3 install tensorflow-gpu

# install Scikit-learn
RUN pip3 install scikit-learn

# install Theano
RUN pip3 install theano

# install Keras
RUN pip3 install keras

# install XGBoost
#RUN git clone --recursive https://github.com/dmlc/xgboost && \
#    mkdir xgboost/build && cd xgboost/build && \
#    cmake .. -DUSE_CUDA=ON && make -j && \
#    cd ../python-package && python3 setup.py install

# configure juypter notebook
#RUN pip3 install jupyter
#COPY jupyter_notebook_config.py /root/.jupyter/

WORKDIR /app
COPY . /app/

# Define default command.
CMD ["bash"]

EXPOSE 12345
