from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from util.dataset import generate_train_val_indices, load_train_val_indices


# Training settings
parser = argparse.ArgumentParser(
    description='CIFAR-10 validation set generator')

parser.add_argument('size', type=int,
                    help='Validation set size')
parser.add_argument('output_file', default='cifar10_split.json',
                    help='Output file (a json file)')




def check():
    args = parser.parse_args()
    train, val = generate_train_val_indices(50000, args.size, args.output_file)
    file_train, file_val = load_train_val_indices(args.output_file)
    print("Train indices matches {}".format(train==file_train))
    print("Val indices matches {}".format(val==file_val))


def main():
    args = parser.parse_args()
    generate_train_val_indices(50000, args.size, args.output_file)


if __name__ == '__main__':
    main()
    #check()
