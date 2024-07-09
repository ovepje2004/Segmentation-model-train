conda create -n detectron_env python=3.8

conda init

conda activate detectron_env

conda update —all

python -m pip install --upgrade pip

nvcc --version

conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch - c nvidia

pip install cython

conda install -c ananconda git

conda install ninja

git clone https://github.com/facebookresearch/detectron2.git

pip install -e detectron2

conda install  -c conda-forge opencv

conda install ipykernel

python -m ipykernel install -user -name detectron_env -display-name detectron_env
