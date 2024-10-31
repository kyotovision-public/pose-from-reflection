import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
from .rm_net import UNet
from .rmap_utils import create_normal_grid


class ReflectanceMapFeatureExtractor(nn.Module):
    def __init__(self, num_fea_mid=36, num_fea_out=36):
        super(ReflectanceMapFeatureExtractor, self).__init__()
        self.unet1 = nn.Sequential(
            UNet(num_fea_in=3+3+3+1, num_fea_out=num_fea_mid),
        )
        self.unet2 = nn.Sequential(
            UNet(num_fea_in=3+3+3+1+num_fea_mid, num_fea_out=num_fea_out),
        )

    def forward(self, rmap):
        bs,c,h,w = rmap.size()
        device = rmap.device

        nx,ny,nz = create_normal_grid(w, projection_mode='probe').to(device).unbind(-1)
        mask = ((nx**2 + ny**2 + nz**2) > 0.0).float()[None,None,:,:]
        rmap_normal = torch.stack([nx,ny,nz], dim=0)[None].repeat(bs,1,1,1)
        solid_angle = (2 * np.pi)**2 / (h * w) * torch.sinc(torch.acos(torch.clamp(nz, -0.99999, 0.99999)) / np.pi).view(1,1,h,w).repeat(bs,1,1,1)
        solid_angle = solid_angle * mask

        energy = torch.sum(rmap**2 * solid_angle, dim=(2,3)) / (torch.sum(solid_angle, dim=(2,3)) + 1e-20)
        mean_color = torch.sqrt(energy + 1e-10)
        log_rmap = torch.log1p(torch.clamp(rmap / mean_color[:,:,None,None],0,None))

        pad = (rmap.size(-1)//4, rmap.size(-1)//4, rmap.size(-1)//4, rmap.size(-1)//4)
        log_rmap_ = F.pad(F.interpolate(log_rmap, size=rmap.size(-1)//2), pad, 'constant', 0)

        # estimate diffuse rmap as sh coefs
        x = torch.cat([
            log_rmap, 
            log_rmap_, 
            rmap_normal, 
            solid_angle
        ], dim=1)
        y1 = self.unet1(x) # bs,24,h,w
        y2 = self.unet2(torch.cat([x,y1], dim=1)) # (bs,c,h,w)

        y2 = y2 / torch.sqrt(torch.sum(y2**2, dim=1, keepdim=True) + 1e-4)

        y2 = y2 * mask

        return y2
