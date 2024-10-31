import torch
import torch.nn as nn
import torch.nn.functional as F

from .rm_net import UNet

class NormalMapFeatureExtractor(nn.Module):
    def __init__(self, num_fea_mid=36, num_fea_out=36):
        super(NormalMapFeatureExtractor, self).__init__()
        self.unet1 = UNet(3,num_fea_mid)
        self.unet2 = UNet(3+num_fea_mid,num_fea_out)

    def forward(self, normal_map):
        #bs,c,h,w = normal_map.size()
        #device = normal_map.device

        x1 = self.unet1(normal_map)
        x2 = self.unet2(
            torch.cat([normal_map, x1], dim=1)
        )

        x2 = x2 / torch.sqrt(torch.sum(x2**2, dim=1, keepdim=True) + 1e-4)

        #x2 = x2 * mask

        return x2
