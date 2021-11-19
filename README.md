# Iterative Training: Finding Binary Weight Deep Neural Networks with Layer Binarization

This repository contains the source code for the paper: [https://arxiv.org/abs/2111.07046](https://arxiv.org/abs/2111.07046).

## Requirements

* GPU
* Python 3
* PyTorch 1.9
  * Earlier version may work, but untested.
* `pip install -r requirements.txt`
* If running ResNet-21 or ImageNet experiments, first download and prepare the ImageNet 2012 dataset with [bin/imagenet_prep.sh](bin/imagenet_prep.sh) script.


## Running

For non-ImageNet experiments, the main python file is [main.py](main.py). To see its arguments:

```sh
python main.py --help
```

Running for the first time can take a little longer due to automatic downloading of the MNIST and Cifar-10 dataset from the Internet.


For ImageNet experiments, the main python files are [main_imagenet_float.py](main_imagenet_float.py) and [main_imagenet_binary.py](main_imagenet_binary.py).
Too see their arguments:

```sh
python main_imagenet_float.py --help
```

and

```sh
python main_imagenet_binary.py --help
```

The ImageNet dataset must be already downloaded and prepared. Please see the requirements section for details.


## Scripts

The main python file has many options. The following scripts runs training with hyper-parameters given in the paper.
Output includes a run-log text file and tensorboard files.
These files are saved to `./logs` and reused for subsequent runs.


### 300-100-10

#### Sensitivity Pre-training

```sh
# Layer 1. Learning rate 0.1.
./scripts/mnist/300/sensitivity/layer.sh sensitivity forward 0.1 0
# Layer 2. Learning rate 0.1.
./scripts/mnist/300/sensitivity/layer.sh sensitivity 231 0.1 0
# Layer 3. Learning rate 0.1.
./scripts/mnist/300/sensitivity/layer.sh sensitivity reverse 0.1 0
```

Output files and run-log are written to `./logs/mnist/val/sensitivity/`.


#### Hyperparam search

For floating-point training:

```sh
# Learning rate 0.1.
./scripts/mnist/300/val/float.sh hyperparam 0.1 0
```

For full binary training:

```sh
# Learning rate 0.1.
./scripts/mnist/300/val/binary.sh hyperparam 0.1 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1.
./scripts/mnist/300/val/layer.sh hyperparam forward 0.1 0
# Reverse order. Learning rate 0.1.
./scripts/mnist/300/val/layer.sh hyperparam reverse 0.1 0
# 1, 3, 2 order. Learning rate 0.1.
./scripts/mnist/300/val/layer.sh hyperparam 132 0.1 0
# 2, 1, 3 order. Learning rate 0.1.
./scripts/mnist/300/val/layer.sh hyperparam 213 0.1 0
# 2, 3, 1 order. Learning rate 0.1.
./scripts/mnist/300/val/layer.sh hyperparam 231 0.1 0
# 3, 1, 2 order. Learning rate 0.1.
./scripts/mnist/300/val/layer.sh hyperparam 312 0.1 0
```

Output files and run-log are written to `./logs/mnist/val/hyperparam/`.


#### Full Training

For floating-point training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/float.sh full 0.1 316 0
```

For full binary training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/binary.sh full 0.1 316 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/layer.sh full forward 0.1 316 0
# Reverse order. Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/layer.sh full reverse 0.1 316 0
# 1, 3, 2 order. Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/layer.sh full 132 0.1 316 0
# 2, 1, 3 order. Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/layer.sh full 213 0.1 316 0
# 2, 3, 1 order. Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/layer.sh full 231 0.1 316 0
# 3, 1, 2 order. Learning rate 0.1. Seed 316.
./scripts/mnist/300/run/layer.sh full 312 0.1 316 0
```

Output files and run-log are written to `./logs/mnist/run/full/`.


### 784-100-10

#### Sensitivity Pre-training

```sh
# Layer 1. Learning rate 0.1.
./scripts/mnist/784/sensitivity/layer.sh sensitivity forward 0.1 0
# Layer 2. Learning rate 0.1.
./scripts/mnist/784/sensitivity/layer.sh sensitivity 231 0.1 0
# Layer 3. Learning rate 0.1.
./scripts/mnist/784/sensitivity/layer.sh sensitivity reverse 0.1 0
```

Output files and run-log are written to `./logs/mnist/val/sensitivity/`.


#### Hyperparam search

For floating-point training:

```sh
# Learning rate 0.1.
./scripts/mnist/784/val/float.sh hyperparam 0.1 0
```

For full binary training:

```sh
# Learning rate 0.1.
./scripts/mnist/784/val/binary.sh hyperparam 0.1 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1.
./scripts/mnist/784/val/layer.sh hyperparam forward 0.1 0
# Reverse order. Learning rate 0.1.
./scripts/mnist/784/val/layer.sh hyperparam reverse 0.1 0
# 1, 3, 2 order. Learning rate 0.1.
./scripts/mnist/784/val/layer.sh hyperparam 132 0.1 0
# 2, 1, 3 order. Learning rate 0.1.
./scripts/mnist/784/val/layer.sh hyperparam 213 0.1 0
# 2, 3, 1 order. Learning rate 0.1.
./scripts/mnist/784/val/layer.sh hyperparam 231 0.1 0
# 3, 1, 2 order. Learning rate 0.1.
./scripts/mnist/784/val/layer.sh hyperparam 312 0.1 0
```

Output files and run-log are written to `./logs/mnist/val/hyperparam/`.


#### Full Training

For floating-point training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/float.sh full 0.1 316 0
```

For full binary training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/binary.sh full 0.1 316 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/layer.sh full forward 0.1 316 0
# Reverse order. Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/layer.sh full reverse 0.1 316 0
# 1, 3, 2 order. Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/layer.sh full 132 0.1 316 0
# 2, 1, 3 order. Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/layer.sh full 213 0.1 316 0
# 2, 3, 1 order. Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/layer.sh full 231 0.1 316 0
# 3, 1, 2 order. Learning rate 0.1. Seed 316.
./scripts/mnist/784/run/layer.sh full 312 0.1 316 0
```

Output files and run-log are written to `./logs/mnist/run/full/`.


### Vgg-5


#### Sensitivity Pre-training

```sh
# Layer 1. Learning rate 0.1.
./scripts/cifar10/vgg5/sensitivity/layer.sh sensitivity 1 0.1 0
# Layer 2. Learning rate 0.1.
./scripts/cifar10/vgg5/sensitivity/layer.sh sensitivity 2 0.1 0
# Layer 5. Learning rate 0.1.
./scripts/cifar10/vgg5/sensitivity/layer.sh sensitivity 5 0.1 0
```

Output files and run-log are written to `./logs/cifar10/val/sensitivity/`.

#### Hyperparam Search

For floating-point training:

```sh
# Learning rate 0.1.
./scripts/cifar10/vgg5/val/float.sh hyperparam 0.1 0
```

For full binary training:

```sh
# Learning rate 0.1.
./scripts/cifar10/vgg5/val/binary.sh hyperparam 0.1 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1.
./scripts/cifar10/vgg5/val/layer.sh hyperparam forward 0.1 0
# Ascend order. Learning rate 0.1.
./scripts/cifar10/vgg5/val/layer.sh hyperparam ascend 0.1 0
# Reverse order. Learning rate 0.1.
./scripts/cifar10/vgg5/val/layer.sh hyperparam reverse 0.1 0
# Descend order. Learning rate 0.1.
./scripts/cifar10/vgg5/val/layer.sh hyperparam descend 0.1 0
# Random order. Learning rate 0.1.
./scripts/cifar10/vgg5/val/layer.sh hyperparam random 0.1 0
```

Output files and run-log are written to `./logs/cifar10/val/hyperparam/`.

#### Full Training

For floating-point training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/float.sh full 0.1 316 0
```

For full binary training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/binary.sh full 0.1 316 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/layer.sh full forward 0.1 316 0
# Ascend order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/layer.sh full ascend 0.1 316 0
# Reverse order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/layer.sh full reverse 0.1 316 0
# Descend order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/layer.sh full descend 0.1 316 0
# Random order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg5/run/layer.sh full random 0.1 316 0
```

Output files and run-log are written to `./logs/cifar10/run/full/`.


### Vgg-9


#### Sensitivity Pre-training

```sh
# Layer 1. Learning rate 0.1.
./scripts/cifar10/vgg9/sensitivity/layer.sh sensitivity 1 0.1 0
# Layer 2. Learning rate 0.1.
./scripts/cifar10/vgg9/sensitivity/layer.sh sensitivity 2 0.1 0
# Layer 5. Learning rate 0.1.
./scripts/cifar10/vgg9/sensitivity/layer.sh sensitivity 5 0.1 0
```

Output files and run-log are written to `./logs/cifar10/val/sensitivity/`.

#### Hyperparam Search

For floating-point training:

```sh
# Learning rate 0.1.
./scripts/cifar10/vgg9/val/float.sh hyperparam 0.1 0
```

For full binary training:

```sh
# Learning rate 0.1.
./scripts/cifar10/vgg9/val/binary.sh hyperparam 0.1 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1.
./scripts/cifar10/vgg9/val/layer.sh hyperparam forward 0.1 0
# Ascend order. Learning rate 0.1.
./scripts/cifar10/vgg9/val/layer.sh hyperparam ascend 0.1 0
# Reverse order. Learning rate 0.1.
./scripts/cifar10/vgg9/val/layer.sh hyperparam reverse 0.1 0
# Descend order. Learning rate 0.1.
./scripts/cifar10/vgg9/val/layer.sh hyperparam descend 0.1 0
# Random order. Learning rate 0.1.
./scripts/cifar10/vgg9/val/layer.sh hyperparam random 0.1 0
```

Output files and run-log are written to `./logs/cifar10/val/hyperparam/`.

#### Full Training

For floating-point training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/float.sh full 0.1 316 0
```

For full binary training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/binary.sh full 0.1 316 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/layer.sh full forward 0.1 316 0
# Ascend order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/layer.sh full ascend 0.1 316 0
# Reverse order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/layer.sh full reverse 0.1 316 0
# Descend order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/layer.sh full descend 0.1 316 0
# Random order. Learning rate 0.1. Seed 316.
./scripts/cifar10/vgg9/run/layer.sh full random 0.1 316 0
```

Output files and run-log are written to `./logs/cifar10/run/full/`.


### ResNet-20


#### Sensitivity Pre-training

```sh
# Layer 1. Learning rate 0.1.
./scripts/cifar10/resnet20/sensitivity/layer.sh sensitivity 1 0.1 0
# Layer 2. Learning rate 0.1.
./scripts/cifar10/resnet20/sensitivity/layer.sh sensitivity 2 0.1 0
# ...
# Layer 20. Learning rate 0.1.
./scripts/cifar10/resnet20/sensitivity/layer.sh sensitivity 20 0.1 0
```

Output files and run-log are written to `./logs/cifar10/val/sensitivity/`.

#### Hyperparam Search

For floating-point training:

```sh
# Learning rate 0.1
./scripts/cifar10/resnet20/val/float.sh hyperparam 0.1 0
```

For full binary training:

```sh
# Learning rate 0.1
./scripts/cifar10/resnet20/val/binary.sh hyperparam 0.1 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1
./scripts/cifar10/resnet20/val/layer.sh hyperparam forward 0.1 0
# Ascend order. Learning rate 0.1
./scripts/cifar10/resnet20/val/layer.sh hyperparam ascend 0.1 0
# Reverse order. Learning rate 0.1
./scripts/cifar10/resnet20/val/layer.sh hyperparam reverse 0.1 0
# Descend order. Learning rate 0.1
./scripts/cifar10/resnet20/val/layer.sh hyperparam descend 0.1 0
# Random order. Learning rate 0.1
./scripts/cifar10/resnet20/val/layer.sh hyperparam random 0.1 0
```

Output files and run-log are written to `./logs/cifar10/val/hyperparam/`.


#### Full Training

For floating-point training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/float.sh full 0.1 316 0
```

For full binary training:

```sh
# Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/binary.sh full 0.1 316 0
```

For iterative training:

```sh
# Forward order. Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/layer.sh full forward 0.1 316 0
# Ascend order. Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/layer.sh full ascend 0.1 316 0
# Reverse order. Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/layer.sh full reverse 0.1 316 0
# Descend order. Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/layer.sh full descend 0.1 316 0
# Random order. Learning rate 0.1. Seed 316.
./scripts/cifar10/resnet20/run/layer.sh full random 0.1 316 0
```

Output files and run-log are written to `./logs/cifar10/run/full/`.


### ResNet-21

To run experiments for ResNet-21, first download and prepare the ImageNet dataset.
See the requirements section at the beginning of this readme.
We assume the dataset is prepared and is at `./imagenet`.


#### Sensitivity Pre-training

```sh
# Layer 1. Learning rate 0.01.
./scripts/imagenet/layer.sh sensitivity ./imagenet 20 "[20]" 20 1 0.01
# Layer 2. Learning rate 0.01.
./scripts/imagenet/layer.sh sensitivity ./imagenet 20 "[20]" 20 2 0.01
# Layer 21. Learning rate 0.01.
./scripts/imagenet/layer.sh sensitivity ./imagenet 20 "[20]" 20 21 0.01
```

Output files and run-log are written to `./logs/imagenet/sensitivity/`.

#### Full Training


For floating-point training:

```sh
# Learning rate 0.01.
./scripts/imagenet/float.sh full ./imagenet 67 "[42,57]" 0.01
```

For full binary training:

```sh
# Learning rate 0.01.
./scripts/imagenet/binary.sh full ./imagenet 67 "[42,57]" 0.01
```

For layer-by-layer training:

```sh
# Forward order
./scripts/imagenet/layer.sh full ./imagenet 67 "[42,57]" 2 forward 0.01
# Ascending order
./scripts/imagenet/layer.sh full ./imagenet 67 "[42,57]" 2 ascend 0.01
```


For all scripts, output files and run-log are written to `./logs/imagenet/full/`.


## License

See [LICENSE](LICENSE)

## Contributing

See the [contributing guide](CONTRIBUTING.md) for details of how to participate in development of the module.
