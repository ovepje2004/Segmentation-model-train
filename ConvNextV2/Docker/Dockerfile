FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
apt-get install -y software-properties-common && \
add-apt-repository -y ppa:deadsnakes/ppa && \
apt-get update && \
apt-get install -y python3.8 curl

RUN apt-get install -y python3-pip

RUN mkdir /convnextv2-app
ADD . /convnextv2-app
WORKDIR /convnextv2-app

RUN python3.8 -m pip install --upgrade pip

RUN python3.8 -m pip install setuptools==69.5.1
RUN python3.8 -m pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip install jupyter

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
