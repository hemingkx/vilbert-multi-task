# ViLBERT-multi-task Installing Instruction

This is the Installing Instruction of [ViLBERT-multi-task](https://github.com/facebookresearch/vilbert-multi-task)

My Environment:

- Ubuntu 18.04
- cuda 10.1

I modified some codes in [the original code](https://github.com/facebookresearch/vilbert-multi-task) based on issues mentioned below. You may find it easier to get your own environment settled using codes in this repo.

------

### 1.Intall vilbert-multi-task

#### Install requirements.txt

```shell
pip install -r requirements.txt
```

#### Install pytorch

pytorch-nightly is required in [vqa-maskrcnn-benchmark](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark/-/blob/master/INSTALL.md), but we found it is not a must. You can just install pytorch which corresponds with your cuda.

**Please be aware: Apex with CUDA and C++ extensions is needed in this program.** To install that,  the cuda of your pytorch should be the same to your system defaulted cuda.

You can check your system defaulted cuda by this command:

```
nvcc --version
```

You can find your pytorch installing command [here](https://pytorch.org/get-started/previous-versions/):

```shell
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1
```

#### Install apex

```shell
# install apex
# https://github.com/NVIDIA/apex/issues/1091
git clone https://github.com/NVIDIA/apex.git
cd apex
git reset --hard a651e2c24ecf97cbf367fd3f330df36760e1c597
python setup.py install --cuda_ext --cpp_ext
```

#### vilbert-multi-task setup

vilbert-multi-task setup

```shell
cd vilbert-multi-task
python setup.py develop
```

#### Install jupyter kernel

```shell
conda install ipykernel
python -m ipykernel install --user --name vilbert --display-name vilbert
```

#### Issues Solved

```shell
# ValueError: numpy.ufunc has the wrong size, try recompiling. Expected 192, got 216
pip uninstall numpy
pip uninstall numpy
pip install numpy

# TypeError: Couldn't build proto file into descriptor pool!
## https://github.com/ValvePython/csgo/issues/8
pip uninstall -y protobuf
pip install --no-binary=protobuf protobuf

# TypeError: Conflict register for file "tensorboard/compat/proto/tensor_shape.proto": tensorboard.TensorShapeProto is already defined in file "tensorboardX/src/tensor_shape.proto".
pip install tensorboardX==1.8

#   File "/content/vilbert-multi-task/tools/refer/refer.py", line 49
#    print 'loading dataset %s into memory...' % dataset                                          ^
# SyntaxError: Missing parentheses in call to 'print'. Did you mean print('loading dataset %s into memory...' % dataset)?
## So they're using python2 script :(
### https://github.com/facebookresearch/vilbert-multi-task/issues/33
cd tools/refer
python setup.py install
make
```

### 2.Install vqa-maskrcnn-benchmark

[Official Tutotial](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark/-/blob/master/INSTALL.md)

#### Install dependencies

```shell
pip install yacs
```

#### Install pycocotools

```shell
# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install
```

#### Install maskrcnn-benchmark

Download vqa-maskrcnn-benchmark [here](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark).

```shell
cd vqa-maskrcnn-benchmark
python setup.py build develop

# test in python
import torch
from maskrcnn_benchmark.layers import nms
```

### 3.Solve BERT problem

```shell
# 'BertTokenizer' object has no attribute 'add_special_tokens_single_sentence'
# https://github.com/facebookresearch/vilbert-multi-task/issues/56
pip uninstall pytorch-transformers
pip install pytorch-transformers
```

