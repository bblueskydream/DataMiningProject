import torch
from torch import nn

class CCFNet(nn.Model):
    def __init__(self, model):
        super(CCFNet, self).__init__()
        self.model = model

        return