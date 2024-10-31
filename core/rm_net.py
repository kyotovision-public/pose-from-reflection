import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

from .rmap_utils import create_normal_grid, panorama2rmap, sample_rmap, Im2RMMappingModule

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, padding_mode='zeros'):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class UNet(nn.Module):
    def __init__(self, num_fea_in=4, num_fea_out=32):
        super(UNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, stride=2)
        
        self.conv0 = ConvBnReLU(num_fea_in, 64, bias=True)

        self.conv1 = ConvBnReLU(64, 128, bias=True)
        self.conv2 = ConvBnReLU(128, 128, bias=True)

        self.conv3 = ConvBnReLU(128, 256, bias=True)
        self.conv4 = ConvBnReLU(256, 256, bias=True)

        self.conv5 = ConvBnReLU(256, 512, bias=True)
        self.conv6 = ConvBnReLU(512, 512, bias=True)

        self.conv7 = ConvBnReLU(512, 256, bias=True)

        self.conv9 = ConvBnReLU(512, 128, bias=True)

        self.conv11 = ConvBnReLU(256, 64, bias=True)

        self.conv12 = nn.Conv2d(128, num_fea_out, 3, stride=1, padding=1)
        
    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(self.downsample(conv0)))
        conv4 = self.conv4(self.conv3(self.downsample(conv2)))
        x = self.conv6(self.conv5(self.downsample(conv4)))
        x = torch.cat([conv4, self.upsample(self.conv7(x))], dim=1)
        x = torch.cat([conv2, self.upsample(self.conv9(x))], dim=1)
        x = torch.cat([conv0, self.upsample(self.conv11(x))], dim=1)
        x = self.conv12(x)
        return x

class ReflectanceMapNet(nn.Module):
    def __init__(self, wo_mask=False, wo_img_filtering=False):
        super(ReflectanceMapNet, self).__init__()
        self.wo_mask = wo_mask
        self.wo_img_filtering = wo_img_filtering
        self.init_masknet = nn.Sequential(UNet(6, 1), nn.Sigmoid()) #FeaNet(6, 256+1)
        fea_dim = 32 if (not self.wo_img_filtering) else 6
        self.feanet = UNet(6, fea_dim) #FeaNet(6, 256+1)
        self.masknet = nn.Sequential(UNet(9, 1), nn.Sigmoid()) #FeaNet(6, 256+1)
        self.rm_decoder_1 = UNet(fea_dim+3+2+3, 3)
        self.rm_decoder_2 = UNet(fea_dim+3+2+3+3, 3)
        self.mapping_module = Im2RMMappingModule()

        rmap_mask = (create_normal_grid((256,256), projection_mode='probe')[...,2] > 0).float()[None,None,:,:]
        self.register_buffer('rmap_mask', rmap_mask)

    def forward(self, img, normal, num_itrs=3):
        img = torch.clamp(img, 0, None)
        mask = torch.any(img > 0, dim=1, keepdim=True).float()
        bs,ch,h,w = img.size()
        device = img.device
        if self.wo_mask:
            num_itrs = 1

        # normalize scale
        mean_color = torch.sum(img * mask, dim=(2,3)) / (torch.sum(mask, dim=(2,3)) + 1e-20)
        img_ = img / mean_color[:,:,None,None]

        assert torch.all(torch.isnan(mean_color) == False)
        assert torch.all(torch.isinf(mean_color) == False)

        # tone mapping
        img_ = torch.log1p(torch.clamp(img_,0,None))

        assert torch.all(torch.isnan(img_) == False)
        assert torch.all(torch.isinf(img_) == False)

        # initial mask estimation
        x_in = torch.cat([img_, normal], dim=1)
        if self.wo_mask:
            est_mask = mask
        else:
            est_mask = torch.clamp(self.init_masknet(x_in) * mask,0,1)

        assert torch.all(torch.isnan(est_mask) == False)
        assert torch.all(torch.isinf(est_mask) == False)

        # feature extraction
        if self.wo_img_filtering:
            img_fea = x_in
        else:
            img_fea = self.feanet(x_in)
        assert torch.all(torch.isnan(img_fea) == False)
        assert torch.all(torch.isinf(img_fea) == False)

        results = []
        for _ in range(num_itrs):
            # mapping
            fea_rmap = self.mapping_module(
                torch.cat([img_fea, img_], dim=1),
                normal,
                est_mask,
                projection_mode='probe',
                out_size=(64,64),
                use_cosine_map=True,
                use_var_map=True,
                hemispherical=True
            )

            fea_rmap = F.interpolate(fea_rmap, scale_factor=4, mode='bilinear')
            #rm_mask = F.interpolate(rm_mask, scale_factor=4, mode='bilinear')
            assert torch.all(torch.isnan(fea_rmap) == False)
            assert torch.all(torch.isinf(fea_rmap) == False)

            est_rmap_ = self.rm_decoder_1(fea_rmap)
            assert torch.all(torch.isnan(est_rmap_) == False)
            assert torch.all(torch.isinf(est_rmap_) == False)
            est_rmap_wo_mask = torch.clamp(self.rm_decoder_2(torch.cat([fea_rmap, est_rmap_], dim=1)), None, 30).exp()
            est_rmap_wo_mask = est_rmap_wo_mask * mean_color[:,:,None,None]
            assert torch.all(torch.isnan(est_rmap_wo_mask) == False)
            assert torch.all(torch.isinf(est_rmap_wo_mask) == False)
            est_rmap = est_rmap_wo_mask * self.rmap_mask
            assert torch.all(torch.isnan(est_rmap) == False)
            assert torch.all(torch.isinf(est_rmap) == False)

            interp_rmap = torch.clamp(fea_rmap[:,-3:], 0, 30).expm1() * mean_color[:,:,None,None]
            assert torch.all(torch.isnan(interp_rmap) == False)
            assert torch.all(torch.isinf(interp_rmap) == False)

            est_img = torch.clamp(sample_rmap(est_rmap_wo_mask, normal, projection_mode='probe'), 0, None)
            est_img_ = torch.log1p(torch.clamp(est_img / mean_color[:,:,None,None],0,None))
            assert torch.all(torch.isnan(est_img) == False)
            assert torch.all(torch.isinf(est_img) == False)
            assert torch.all(torch.isnan(est_img_) == False)
            assert torch.all(torch.isinf(est_img_) == False)

            if not self.wo_mask:
                est_mask = torch.clamp(self.masknet(
                    torch.cat([img_, normal, est_img_], dim=1)
                ) * mask, 0, 1)
            assert torch.all(torch.isnan(est_mask) == False)
            assert torch.all(torch.isinf(est_mask) == False)

            if False:
                from .sfs_utils import plot_hdr, plot_normal_map
                import matplotlib.pyplot as plt
                plt.subplot(2,6,1)
                plot_hdr(img / torch.max(img.view(bs,-1), dim=1)[0][:,None,None,None])
                plt.subplot(2,6,2)
                plot_normal_map(normal)
                plt.subplot(2,6,3)
                plt.imshow(est_mask[0,0].detach().cpu())
                plt.subplot(2,6,4)
                plot_hdr(interp_rmap / torch.max(img.view(bs,-1), dim=1)[0][:,None,None,None])
                plt.subplot(2,6,5)
                plt.imshow(rm_mask[0,0].detach().cpu())
                plt.subplot(2,6,6)
                plot_hdr(est_rmap / torch.max(img.view(bs,-1), dim=1)[0][:,None,None,None])

                plt.subplot(2,6,7)
                plot_hdr(est_img / torch.max(img.view(bs,-1), dim=1)[0][:,None,None,None])
                plt.show()



            result = {
                'rmap': est_rmap,
                'rmap_wo_mask': est_rmap_wo_mask,
                'fea_rmap': fea_rmap,
                'interp_rmap': interp_rmap,
                'est_img': est_img,
                'est_mask': est_mask,
            }
            results.append(result)

        return results