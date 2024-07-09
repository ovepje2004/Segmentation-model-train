apt-get update

apt-get install -y python3.8 python3.8-dev python3.8-venv

update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

apt-get update

apt-get install -y libblas-dev liblapack-dev

python3 -m pip install packaging

python3.8 -m pip install setuptools wheel

python3 -m pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --no-build-isolation ./

cd /usr/local/lib/python3.8/dist-packages

apt-get update

apt-get install vim

vi helpers.py

import timm -> import collections.abc as container_abcs

pip install --upgrade timm

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

