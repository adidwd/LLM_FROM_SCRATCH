import os
import torch
from torch import nn
import sys
sys.path.append('./')
class DNN(nn.Module):

    def __init__(self,layer_sizes,use_shortcut):
        super().__init__()
        self.use_shortcut=use_shortcut
        self.layers=nn.Module()