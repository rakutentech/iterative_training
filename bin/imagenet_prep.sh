#!/bin/bash
#
# This script prepares the original ImageNet 2012 tar files for Pytorch's datasets.ImageFolder.
# The download is about 150GB and will be double that size when preparation is done.
#
# Download these files from the ImageNet website (https://image-net.org):
# $ ls
# ILSVRC2012_devkit_t12.tar.gz  ILSVRC2012_img_train.tar  ILSVRC2012_img_val.tar
#
# When this script is finished:
# $ ls
# imagenet  ILSVRC2012_devkit_t12.tar.gz  ILSVRC2012_img_train.tar  ILSVRC2012_img_val.tar
# $ ls imagenet
# ILSVRC2012_devkit_t12.tar.gz  train  val
#
# Run time
#real    16m33.078s
#user    2m10.273s
#sys     9m24.362s


mkdir -p imagenet && cd imagenet
cp ../ILSVRC2012_devkit_t12.tar.gz .

# Extract the training data
mkdir train && cd train
tar -xvf ../ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

# Extract the validation data and move images to subfolders:
mkdir val && cd val
tar -xvf ../ILSVRC2012_img_val.tar
wget -qO- "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh" | bash
