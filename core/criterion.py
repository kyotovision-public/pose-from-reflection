import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def gauss_kernel(size=5, device=torch.device('cuda:0'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel.cuda(), groups=img.shape[1])
    return out

def compute_multiscale_loss(rendered_img, gt_img, num_scales=3, scale_factor=1, beta=1e-3):
    def compute_loss(a,b):
        return F.smooth_l1_loss(a,b,beta=beta)
        return torch.mean(torch.abs(a-b)**pow)
    total_loss = 0.0
    total_weight = 0.0
    kernel = gauss_kernel()
    weight_per_scale = 1.0
    for _ in range(num_scales):
        total_loss = total_loss + weight_per_scale * compute_loss(rendered_img, gt_img)
        total_weight = total_weight + weight_per_scale

        rendered_img_filtered = conv_gauss(rendered_img, kernel)
        gt_img_filtered = conv_gauss(gt_img, kernel)

        rendered_img = rendered_img_filtered[:,:,::2,::2]
        gt_img = gt_img_filtered[:,:,::2,::2]

        weight_per_scale *= scale_factor**2
    return total_loss / total_weight

def compute_mask_loss(est_mask, gt_mask, num_scales=3, scale_factor=0.25):
    return compute_multiscale_loss(est_mask, gt_mask, num_scales=4, beta=1)

class ImageLogL1Loss(torch.nn.Module):
    def __init__(self, pow=1):
        super(ImageLogL1Loss, self).__init__()
        self.pow = pow

    def forward(self, hdr_image_a, hdr_image_b, mask=None, beta=1e-3, num_scales=3):
        max_radiance = torch.max(hdr_image_b.detach().contiguous().view(hdr_image_b.size(0),-1), dim=1)[0]
        #max_radiance = torch.max(torch.cat([hdr_image_a, hdr_image_b], dim=1).view(hdr_image_a.size(0),-1), dim=1)[0]
        # tone mapping
        def tone_mapping(hdr_image):
            hdr_image = hdr_image# / max_radiance[:,None,None,None]
            hdr_image = torch.clamp(hdr_image, 0, None)
            sdr_image = torch.log1p(1 * hdr_image) / np.log1p(1 * 1)
            #hdr_image = torch.clamp(hdr_image, 0, None)
            return sdr_image#torch.log1p(hdr_image)

        sdr_image_a = tone_mapping(hdr_image_a)
        sdr_image_b = tone_mapping(hdr_image_b)

        if not (mask is None):
            sdr_image_a = sdr_image_a * mask
            sdr_image_b = sdr_image_b * mask

        return compute_multiscale_loss(sdr_image_a, sdr_image_b, beta=beta, num_scales=num_scales)

class ImageGradientLoss(torch.nn.Module):
    def __init__(self):
        super(ImageGradientLoss, self).__init__()
        self.filter = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=2, 
            kernel_size=3, 
            stride=1, 
            padding=0,
            bias=False
        )
        self.filter.weight.requires_grad = False
        self.filter.weight[:] = 0.0

        self.filter.weight[0,0,0,0] = self.filter.weight[0,0,2,0] = -0.125
        self.filter.weight[0,0,1,0] = -0.25
        self.filter.weight[0,0,0,2] = self.filter.weight[0,0,2,2] = 0.125
        self.filter.weight[0,0,1,2] = 0.25

        self.filter.weight[1,0,0,0] = self.filter.weight[1,0,0,2] = -0.125
        self.filter.weight[1,0,0,1] = -0.25
        self.filter.weight[1,0,2,0] = self.filter.weight[1,0,2,2] = 0.125
        self.filter.weight[1,0,2,1] = 0.25

    def forward(self, hdr_image_a, hdr_image_b, beta=1e-3):
        # tone mapping
        def tone_mapping(hdr_image):
            hdr_image = hdr_image# / max_radiance[:,None,None,None]
            hdr_image = torch.clamp(hdr_image, 0, None)
            sdr_image = torch.log1p(1 * hdr_image) / np.log1p(1 * 1)
            #hdr_image = torch.clamp(hdr_image, 0, None)
            return sdr_image#torch.log1p(hdr_image)

        sdr_image_a = tone_mapping(hdr_image_a)
        sdr_image_b = tone_mapping(hdr_image_b)

        def apply_sobel(img):
            return torch.stack([self.filter(im[:,None]) for im in img.unbind(1)], dim=2)
        
        grad_a = apply_sobel(sdr_image_a)
        grad_b = apply_sobel(sdr_image_b)

        return compute_multiscale_loss(grad_a[:,0], grad_b[:,0], beta=beta) + compute_multiscale_loss(grad_a[:,1], grad_b[:,1], beta=beta)

class OccludingBoundaryLoss(torch.nn.Module):
    def __init__(self):
        super(OccludingBoundaryLoss, self).__init__()
        self.filter = torch.nn.Conv2d(
            in_channels=1, 
            out_channels=2, 
            kernel_size=3, 
            stride=1, 
            padding=1,
            bias=False
        )
        self.filter.weight.requires_grad = False
        self.filter.weight[:] = 0.0

        self.filter.weight[0,0,0,0] = self.filter.weight[0,0,2,0] = 0.125
        self.filter.weight[0,0,1,0] = 0.25
        self.filter.weight[0,0,0,2] = self.filter.weight[0,0,2,2] = -0.125
        self.filter.weight[0,0,1,2] = -0.25

        self.filter.weight[1,0,0,0] = self.filter.weight[1,0,0,2] = -0.125
        self.filter.weight[1,0,0,1] = -0.25
        self.filter.weight[1,0,2,0] = self.filter.weight[1,0,2,2] = 0.125
        self.filter.weight[1,0,2,1] = 0.25

    def forward(self, normal_map, mask):
        mask_grad = self.filter(mask)
        mask_grad = mask_grad / torch.sqrt(torch.sum(mask_grad**2,dim=1,keepdim=True)+1e-6)

        mask_ = (torch.sum(mask_grad**2,dim=1,keepdim=True) > 0.0).float()
        mask_ *= (torch.sum(normal_map.detach()**2,dim=1,keepdim=True) > 0.0).float()

        error_map = torch.acos(torch.clamp(torch.sum(normal_map[:,:2] * mask_grad, dim=1, keepdim=True),-0.99999,0.99999)) * mask_

        return torch.mean(torch.sum(error_map * mask_, dim=(2,3)) / torch.sum(mask_, dim=(2,3)))


class VGG16PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGG16PerceptualLoss, self).__init__()
        if not os.path.exists('./pretrained-models/vgg16/vgg16.pt'):
            import subprocess
            subprocess.run(['wget', '-P', './pretrained-models/vgg16', 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'])
        self.vgg16 = torch.jit.load('./pretrained-models/vgg16/vgg16.pt').eval()

    def forward(self, hdr_image_a, hdr_image_b):
        self.vgg16 = self.vgg16.to(hdr_image_a.device)
        max_radiance = torch.max(hdr_image_b.view(hdr_image_a.size(0),-1), dim=1)[0]

        # tone mapping
        def tone_mapping(hdr_image):
            hdr_image = hdr_image / max_radiance[:,None,None,None]
            hdr_image = torch.clamp(hdr_image, 0, None)
            sdr_image = torch.log1p(100 * hdr_image) / np.log1p(100 * 1)
            return 255 * sdr_image
        sdr_image_a = tone_mapping(hdr_image_a)
        sdr_image_b = tone_mapping(hdr_image_b)

        # feature extraction
        def extract_fea(sdr_image):
            if sdr_image.size(-1) > 256:
                sdr_image = F.interpolate(sdr_image, size=(256, 256), mode='area')
            return self.vgg16(sdr_image, resize_images=False, return_lpips=True)
        fea_a = extract_fea(sdr_image_a)
        fea_b = extract_fea(sdr_image_b)

        return torch.mean(torch.sum((fea_a - fea_b)**2, dim=1))

