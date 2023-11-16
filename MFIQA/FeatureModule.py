import torch
import torch.nn as nn
import torch.nn.init as init
from module.dat_mod import DAT_mod
from module.mod_MAN import mod_MAN
from module.Sam_module import Sam_module

class FeatureModule(nn.Module):
    def __init__(self):
        super(FeatureModule, self).__init__()
        self.dat_mod = DAT_mod()
        self.mod_MAN = mod_MAN()
        self.Sam_module = Sam_module()

    def forward(self, x):
        x1, x2, x3 = x
        fea1 = self.dat_mod(x1)
        fea2 = self.mod_MAN(x2)
        fea3 = self.Sam_module(x3)

        return fea1, fea2, fea3

