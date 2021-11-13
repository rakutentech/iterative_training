from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from .binary_control import ControlBinaryLinear, ControlBinaryConv2d

class LayerControl:
    """Layer control for the main model class.
    """

    def layer_init(self, 
            epochs_per_layer : int, 
            all_binary : bool, 
            layers_json : str,
    ) -> None:
        self.all_binary_ = all_binary
        # Track training progress
        self.epochs_per_layer_ = epochs_per_layer
        self.epochs_ = 0
        self.nets_ = list(self.modules())
        self.layers_ = self.load_layers(layers_json)
        if self.layers_ is None:
            raise ValueError(f"Problem with reading layers json file: {layers_json}")
        self.sanity_check()
        if all_binary:
            print('layer_init() binarize_all()')
            self.binarize_all()

    def sanity_check(self):
        # A layer that can be binarized have this variable
        for n in self.layers_:
            assert hasattr(self.nets_[n], "binarize_")

    def load_layers(self, json_file):
        print(f"Loading layers from {json_file}...")
        with open(json_file, 'r') as fp:
            return json.load(fp)
        return None

    def detect_layers(self) -> list:
        """Return layer indices of binarizable neural network
        """
        layers = []
        test = (ControlBinaryLinear, ControlBinaryConv2d)
        for idx, m in enumerate(self.modules()):
            if isinstance(m, test):
                layers.append(idx)
                #print(idx, '->', m)
        #print(len(layers), "layers", layers)
        return layers

    def show_binarized(self):
        status = ""
        layers = ""
        for n in self.layers_:
            status += "{:3}".format(self.nets_[n].binarize_)
            layers += "{:3}".format(n)
        print(f'Epoch: {self.epochs_}')
        print("Layer ", layers)
        print("Binary", status)

    def binarize_layer(self):
        """Set layer binarization on a forward-first schedule.
        Call before training an epoch.
        """
        layer = self.epochs_ // self.epochs_per_layer_
        if layer < len(self.layers_):
            self.nets_[self.layers_[layer]].binarize_ = True
        else:
            print("All already binarized")
        self.epochs_ += 1
        self.show_binarized()

    def binarize_layer_reverse(self):
        """Set layer binarization on a last-first schedule.
        Call before training an epoch.
        """
        layer = self.epochs_ // self.epochs_per_layer_
        if layer < len(self.layers_):
            self.nets_[self.layers_[-1-layer]].binarize_ = True
        else:
            print("All already binarized")
        self.epochs_ += 1
        self.show_binarized()

    def binarize_all(self):
        """Binarize all layers
        Call once before starting any training.
        """
        for n in self.layers_:
            self.nets_[n].binarize_ = True
        self.show_binarized()
