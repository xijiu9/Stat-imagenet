StatQuant
====
Code for reproducing the results in the paper [A Statistical Framework for Low-bitwidth Training of Deep Neural Networks](https://arxiv.org/abs/2010.14298)
published in NeurIPS 2020. 
The code is based [convNet.pytorch](https://github.com/eladhoffer/convNet.pytorch), [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples), and
[pytorch_minimax](https://github.com/GaiYu0/pytorch_minimax/tree/master).

INSTALL
----
Tested with PyTorch 1.4.0 + CUDA 10.1.

Step 1: Install apex
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Step 2: Install this repo
```bash
# Make sure that your nvcc version is compatible with the cuda library version used by PyTorch
nvcc --version
cd pytorch_minimax
python setup.py install
cd ..
cd quantizers
python setup.py install
cd ..
```

CIFAR10
----

```bash
mkdir results
# Exact
./train_cifar preact_resnet56 exact "" 
# QAT
./train_cifar preact_resnet56 qat "-c quantize --qa=True --qw=True --qg=False" 
# 8-bit PTQ
./train_cifar preact_resnet56 ptq_8 "-c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False --bbits=8"
# 8-bit PSQ 
./train_cifar preact_resnet56 psq_8 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits=8" 
# 8-bit BHQ
./train_cifar preact_resnet56 hqq_8 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits=8"   
```

ImageNet
----

Tested with GTX 1080Ti * 8, RTX 2080Ti * 8, and V100 * 8. Please put the ILSVRC2012 dataset at ```<imagenet_path>```. 
```bash
mkdir results
# Exact
./dist-train 1 0 127.0.0.1 8 resnet18 "" exact <imagenet_path>  
# QAT
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=False" qat <imagenet_path>  
# 8-bit PTQ
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=False --hadamard=False --bbits=8" ptq_8 <imagenet_path>
# 8-bit PSQ
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=False --bbits=8" psq_8 <imagenet_path> 
# 8-bit BHQ
./dist-train 1 0 127.0.0.1 8 resnet18 "-c quantize --qa=True --qw=True --qg=True --persample=True --hadamard=True --bbits=8" bhq_8 <imagenet_path>
```

License
----
MIT

Reference
----

If you find this repo useful, please cite our paper.
```
@inproceedings{chen2020statistical,
  title={A Statistical Framework for Low-bitwidth Training of Deep Neural Networks},
  author={Chen, Jianfei and Gai, Yu and Yao, Zhewei and Mahoney, Michael and Gonzalez, Joseph},
  booktitle={Advances in Neural Information Processing Systems},  
  year={2020}
}
```


Some Notes
----

Training Speed
```
resnet18 imagenet       4500ips
resnet50 imagenet amp   3200ips
```
