import torch
import torch.nn as nn
import torch.nn.functional as F

import numba
from numba import cuda
import math
import numpy as np

import ctypes

from .rmap_utils import sample_rmap, create_normal_grid

# reference: https://gist.github.com/t-vi/2f4fe23a5b473b9dceb95b163378b4d5
def as_cuda_array(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.cudadrv.devices.get_context(t.device.index)
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    return cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('float32'), 
                                                  gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)

# log_img: BS,3,H,W
# log_rmap: BS,3,Hn,Wn
# grid: BS,2,N,H,W
# min_error: BS,1,N,H,W
# argmin_normal: BS,3,N,H,W
# path_size: (integer)
@cuda.jit
def find_local_argmin_normal_kernel(log_img, log_rmap, grid, min_error, argmin_normal, patch_size):
    BS,C,Hn,Wn = log_rmap.shape
    N,H,W = grid.shape[2:]

    idx_batch = cuda.blockIdx.x
    idx_v = cuda.blockIdx.y
    idx_u = cuda.blockIdx.z

    idx_thread = cuda.threadIdx.x
    blockdim = cuda.blockDim.x

    log_i = log_img[idx_batch,:,idx_v,idx_u]

    idx_patch = idx_thread
    while idx_patch < N:
        idx_un0 = grid[idx_batch,0,idx_patch,idx_v,idx_u]
        idx_vn0 = grid[idx_batch,1,idx_patch,idx_v,idx_u]
        min_e = 1e9
        argmin_nx = 0.0
        argmin_ny = 0.0
        for idx_vn in range(idx_vn0,idx_vn0+patch_size):
            for idx_un in range(idx_un0,idx_un0+patch_size):
                nx = (idx_un + 0.5) / Wn * 2.0 - 1.0
                ny = -((idx_vn + 0.5) / Hn * 2.0 - 1.0)
                if (nx**2 + ny**2) > 1.0:
                    continue
                e = 0.0
                log_r = log_rmap[idx_batch,:,idx_vn,idx_un]
                is_valid = True
                for idx_ch in range(C):
                    e += abs(log_i[idx_ch] - log_r[idx_ch])
                    is_valid *= (log_r[idx_ch] != 0.0) * (log_i[idx_ch] != 0.0)
                if is_valid and (e < min_e):
                    argmin_nx = nx
                    argmin_ny = ny
                    min_e = e

        if min_e == 1e9:
            argmin_nx = (idx_un0 + 0.5 * patch_size) / Wn * 2.0 - 1.0
            argmin_ny = -((idx_vn0 + 0.5 * patch_size) / Hn * 2.0 - 1.0)
        argmin_nz = math.sqrt(min(max(0.0, 1.0 - argmin_nx**2 - argmin_ny**2), 1.0))

        min_error[idx_batch,0,idx_patch,idx_v,idx_u] = min_e
        argmin_normal[idx_batch,0,idx_patch,idx_v,idx_u] = argmin_nx
        argmin_normal[idx_batch,1,idx_patch,idx_v,idx_u] = argmin_ny
        argmin_normal[idx_batch,2,idx_patch,idx_v,idx_u] = argmin_nz

        idx_patch += blockdim


def find_local_argmin_normal(log_img, log_rmap, grid, patch_size):
    BS = log_rmap.size()[0]
    N,H,W = grid.size()[2:]
    dtype = log_img.dtype
    device = log_img.device

    min_error = torch.zeros((BS,1,N,H,W), dtype=dtype, device=device)
    argmin_normal = torch.zeros((BS,3,N,H,W), dtype=dtype, device=device)

    log_img_n = as_cuda_array(log_img.detach())
    log_rmap_n = as_cuda_array(log_rmap.detach())
    grid_n = cuda.as_cuda_array(grid.detach())
    min_error_n = as_cuda_array(min_error)
    argmin_normal_n = as_cuda_array(argmin_normal)

    num_threads = min(N,256)
    find_local_argmin_normal_kernel[(BS,H,W),(num_threads)](log_img_n, log_rmap_n, grid_n, min_error_n, argmin_normal_n, patch_size)

    return min_error, argmin_normal

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, bias=True):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)

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

def pad_invalid_values(x):
    if x.requires_grad == False:
        mask = (torch.isnan(x) == False)
        mask *= (torch.isinf(x) == False)
        x[mask==False] = 0.0
    return torch.clamp(x, 0.0, 1e24)

class SfSNet(nn.Module):
    def __init__(self, initial_patch_size = 16):
        super(SfSNet, self).__init__()
        self.initial_patch_size = initial_patch_size
        self.likelihood_param = nn.Parameter(torch.randn(1))
        self.feanet = nn.Sequential(
            nn.Conv3d(4,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
        )
        self.feanet_i = UNet(3,16)
        self.aggnet = UNet(64+16,32)
        self.pos_encoder = nn.Sequential(
            nn.Conv3d(3,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,32,1),            
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(32+32,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,64,1),
            nn.LeakyReLU(),
            nn.Conv3d(64,1,1),
            nn.Softplus(),
        )

    # img:  (BS*3*H*W)
    # rmaps: list of (BS*3*H*W)
    def forward(self, img, rmap):
        BS,C,Hn,Wn = rmap.size()
        H,W = img.size()[2:4]
        device = img.device

        img = pad_invalid_values(img)
        rmap = pad_invalid_values(rmap)

        mask = (torch.max(img, dim=1, keepdim=True)[0] > 0.0).float()

        # make mask of rmap
        v,u = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
        x = 2 * (u + 0.5) / Wn - 1
        y = -(2 * (v + 0.5) / Hn - 1)
        z = torch.sqrt(torch.clamp(1-x**2-y**2,0,None))
        rmap_mask = (z > 0.0).float()[None,None].to(rmap.device)

        # scale normalization & apply log function
        mean_color = torch.sum(rmap*rmap_mask, dim=(2,3)) / torch.sum(rmap_mask, dim=(2,3)) # [BS,3]
        log_img = torch.log1p(100.0 * img / mean_color[:,:,None,None]) * mask
        log_rmap = torch.log1p(100.0 * rmap / mean_color[:,:,None,None]) * rmap_mask

        if False:
            log_mean_i = (torch.sum(log_img * mask, dim=(0,2,3)) / torch.sum(mask, dim=(0,2,3)))[None,:,None,None]
            log_mean_r = (torch.sum(log_rmap * rmap_mask, dim=(0,2,3)) / torch.sum(rmap_mask, dim=(0,2,3)))[None,:,None,None]
            log_max_i = torch.max(log_img.view(3,-1), dim=1)[0][None,:,None,None]
            log_max_r = torch.max(log_rmap.view(3,-1), dim=1)[0][None,:,None,None]
            log_rmap = (log_rmap - log_mean_r) / (log_max_r - log_mean_r) * (log_max_i - log_mean_i) + log_mean_i


        fea_img = self.feanet_i(log_img)

        results = []
        patch_size = self.initial_patch_size
        while patch_size >= 4:
            if patch_size == self.initial_patch_size:
                # make initial grid
                Hn_ = Hn // patch_size
                Wn_ = Wn // patch_size
                v_,u_ = torch.meshgrid(torch.arange(Hn_), torch.arange(Wn_))
                u_ = patch_size * u_.to(device)
                v_ = patch_size * v_.to(device)
                grid = torch.stack([u_,v_], dim=0).view(2,-1) # [2,Hn_*Wn_]
                grid = grid[None,:,:,None,None].repeat(BS,1,1,H,W) # [BS,2,Hn_*Wn_,H,W]
            else:
                # dense sampling around good hypotheses
                N = grid.size(2)
                grid_00 = grid[:,:,:N//4]
                grid_01 = grid_00 + torch.tensor([0,patch_size], dtype=grid.dtype, device=device)[None,:,None,None,None]
                grid_10 = grid_00 + torch.tensor([patch_size,0], dtype=grid.dtype, device=device)[None,:,None,None,None]
                grid_11 = grid_00 + torch.tensor([patch_size,patch_size], dtype=grid.dtype, device=device)[None,:,None,None,None]
                grid = torch.cat([grid_00,grid_01,grid_10,grid_11], dim=2)

            # find hypotheses in the patches
            hypotheses_error, hypotheses_normal = find_local_argmin_normal(log_img, log_rmap, grid, patch_size)
            hypotheses_likelihood = self.compute_hypotheses_likelihood(log_img, log_rmap, hypotheses_normal)
            #hypotheses_likelihood = torch.exp(-F.softplus(self.likelihood_param) * hypotheses_error)
            hypotheses_normal = hypotheses_normal * mask[:,:,None,:,:] # [BS,3,N,H,W]
            hypotheses_likelihood = hypotheses_likelihood * mask[:,:,None,:,:] # [BS,1,N,H,W]

            hypotheses = torch.cat([hypotheses_normal, hypotheses_likelihood], dim=1) # [BS,4,N,H,W]

            # per-pixel feature extraction
            fea_sfs = torch.max(self.feanet(hypotheses), dim=2)[0]
            fea_per_pixel = torch.cat([fea_sfs, fea_img], dim=1)

            # spatial aggregation
            fea_agg = self.aggnet(fea_per_pixel)[:,:,None,:,:].repeat(1,1,hypotheses.size(2),1,1)

            # decode aggregated information
            fea_pos = self.pos_encoder(hypotheses_normal)
            fea_per_hypotheses = torch.cat([fea_pos, fea_agg], dim=1)

            # compute final normal probability
            w = self.decoder(fea_per_hypotheses)
            hypotheses_probabililty = w * hypotheses_likelihood
            hypotheses_probabililty = hypotheses_probabililty / torch.clamp(torch.sum(hypotheses_probabililty, dim=2, keepdim=True), 1e-9, None)

            # sort results
            #sorted_indices =torch.argsort(-hypotheses_likelihood, dim=2)
            sorted_indices =torch.argsort(-hypotheses_probabililty, dim=2)
            hypotheses_probabililty = torch.gather(hypotheses_probabililty, dim=2, index=sorted_indices)
            hypotheses_likelihood = torch.gather(hypotheses_likelihood, dim=2, index=sorted_indices)
            hypotheses_normal = torch.gather(hypotheses_normal, dim=2, index=sorted_indices.repeat(1,3,1,1,1))
            grid = torch.gather(grid, dim=2, index=sorted_indices.repeat(1,2,1,1,1))

            grid_normal = 2 * grid.float() / torch.tensor([Wn,Hn], device=device)[None,:,None,None,None] - 1
            grid_normal = grid_normal * torch.tensor([1.0, -1.0], device=device)[None,:,None,None,None]
            normalized_patch_size = 2.0 * patch_size / Hn

            result = {
                'hypotheses_normal': hypotheses_normal, # [BS,3,N,H,W]
                'hypotheses_likelihood': hypotheses_likelihood, # [BS,1,N,H,W]
                'hypotheses_prior': w, # [BS,1,N,H,W]
                'hypotheses_probability': hypotheses_probabililty, # [BS,1,N,H,W]
                'grid_normal': grid_normal, # [BS,2,N,H,W],
                'prior_feature': fea_agg[:,:,0],
                'patch_size': normalized_patch_size
            }
            results.append(result)

            # plot
            if False:
                from .sfs_utils import plot_hdr, plot_normal_map
                import matplotlib.pyplot as plt
                plt.close()
                plt.subplot(1,5,1)
                plot_hdr(img / torch.clamp(torch.max(rmap.view(BS,-1), dim=1)[0], None, 1.0)[:,None,None,None], idx_batch=0)
                plt.subplot(1,5,2)
                plot_hdr(rmap / torch.clamp(torch.max(rmap.view(BS,-1), dim=1)[0], None, 1.0)[:,None,None,None], idx_batch=0)
                plt.subplot(1,5,3)
                plot_normal_map(hypotheses_normal[:,:,0], idx_batch=0)
                g = plt.subplot(1,5,4)
                g.scatter(grid_normal[0,0,:,64,64].cpu(), grid_normal[0,1,:,64,64].cpu())
                g.scatter(hypotheses_normal[0,0,:32,64,64].cpu(), hypotheses_normal[0,1,:32,64,64].cpu())
                g.set_xlim([-1,1])
                g.set_ylim([-1,1])
                g.set_aspect('equal')

                plt.subplot(1,5,5)
                nx = grid_normal[:,0,0]
                ny = grid_normal[:,1,0]
                nz = torch.sqrt(torch.clamp(1 - nx**2 - ny**2, 0, 1))
                plot_normal_map(torch.stack([nx,ny,nz], dim=1) * mask, idx_batch=0)   
                plt.show()      

            patch_size = patch_size // 2

        return results

    def construct_probability_volume(self, img, rmap, chunk_size = 64, stage=-1):
        # get per-pixel latent feature
        result = self.forward(img, rmap)[stage]
        sampled_normals = result['hypotheses_normal']
        fea_agg = result['prior_feature'][:,:,None,:,:]

        BS,C,Hn,Wn = rmap.size()
        H,W = img.size()[2:4]
        device = img.device

        img = pad_invalid_values(img)
        rmap = pad_invalid_values(rmap)

        mask = (torch.max(img, dim=1, keepdim=True)[0] > 0.0).float()

        # make grid and rmap_mask
        v,u = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
        x = 2 * (u + 0.5) / Wn - 1
        y = -(2 * (v + 0.5) / Hn - 1)
        z = torch.sqrt(torch.clamp(1-x**2-y**2,0,None))
        rmap_mask = (z > 0.0).float()[None,None].to(rmap.device)
        grid = torch.stack([u,v], dim=0).view(1,2,Hn*Wn,1,1).repeat(BS,1,1,H,W).to(device)
        rmap_area = (4.0 / (Wn * Hn) / torch.clamp(z[None,None].to(device), 1e-4,None) * rmap_mask)

        # scale normalization & apply log function
        mean_color = torch.sum(rmap*rmap_mask, dim=(2,3)) / torch.sum(rmap_mask, dim=(2,3)) # [BS,3]
        log_img = torch.log1p(100.0 * img / mean_color[:,:,None,None]) * mask
        log_rmap = torch.log1p(100.0 * rmap / mean_color[:,:,None,None]) * rmap_mask

        # compute likelihood
        hypotheses_likelihood_chunks = []
        hypotheses_normal_chunks = []
        for idx_chunk in range((Hn * Wn - 1) // chunk_size + 1):
            grid_chunk = grid[:,:,chunk_size*idx_chunk:chunk_size*(idx_chunk+1)]
            hypotheses_error_chunk, hypotheses_normal_chunk = find_local_argmin_normal(log_img, log_rmap, grid_chunk, 1)
            hypotheses_normal_chunk = hypotheses_normal_chunk * mask[:,:,None,:,:] # [BS,3,N,H,W]
            hypotheses_likelihood_chunk = torch.exp(-F.softplus(self.likelihood_param) * hypotheses_error_chunk) * mask[:,:,None,:,:] # [BS,1,N,H,W]

            hypotheses_likelihood_chunks.append(hypotheses_likelihood_chunk)
            hypotheses_normal_chunks.append(hypotheses_normal_chunk)

        hypotheses_likelihood = torch.cat(hypotheses_likelihood_chunks, dim=2)
        hypotheses_normal = torch.cat(hypotheses_normal_chunks, dim=2)

        # decode aggregated information
        w_chunks = []
        for hypotheses_normal_chunk in hypotheses_normal_chunks:
            fea_pos_chunk = self.pos_encoder(hypotheses_normal_chunk)
            fea_per_hypotheses_chunk = torch.cat([fea_pos_chunk, fea_agg.repeat(1,1,fea_pos_chunk.size(2),1,1)], dim=1)
            w_chunk = self.decoder(fea_per_hypotheses_chunk)
            w_chunks.append(w_chunk)
        w = torch.cat(w_chunks, dim=2)

        # compute final normal probability
        hypotheses_probabililty = w * hypotheses_likelihood
        hypotheses_probabililty = hypotheses_probabililty / torch.clamp(torch.sum(hypotheses_probabililty, dim=2, keepdim=True), 1e-9, None)

        likelihood_volume = hypotheses_likelihood.view(BS,Hn,Wn,H,W)
        prior_volume = w.view(BS,Hn,Wn,H,W)
        posterior_volume = hypotheses_probabililty.view(BS,Hn,Wn,H,W)
        normal_volume = hypotheses_normal.view(BS,3,Hn,Wn,H,W)

        # normalize posterior volume
        posterior_volume = posterior_volume / torch.sum(posterior_volume * rmap_area[:,0,:,:,None,None], dim=(2,3), keepdim=True)

        # compute argmax_normal
        #sorted_indices =torch.argsort(-hypotheses_likelihood, dim=2)
        argmax_indices =torch.max(hypotheses_probabililty, dim=2, keepdim=True)[1]
        pred_normal = torch.gather(hypotheses_normal, dim=2, index=argmax_indices.repeat(1,3,1,1,1))[:,:,0]

        if False:
            import matplotlib.pyplot as plt
            from .sfs_utils import plot_normal_prob, plot_hdr, plot_normal_map
            plt.close()
            plt.subplot(1,5,1)
            plot_hdr(img / torch.clamp(torch.max(rmap.view(BS,-1), dim=1)[0], None, 1.0)[:,None,None,None], idx_batch=0)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(1,5,2)
            plot_hdr(rmap / torch.clamp(torch.max(rmap.view(BS,-1), dim=1)[0], None, 1.0)[:,None,None,None], idx_batch=0)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.subplot(1,5,3)
            plot_normal_prob(likelihood_volume, 0, 64, 64)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.xlabel('likelihood')
            plt.subplot(1,5,4)
            plot_normal_prob(prior_volume, 0, 64, 64)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.xlabel('prior')
            plt.subplot(1,5,5)
            plot_normal_prob(posterior_volume, 0, 64, 64)
            plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False) 
            plt.tick_params(bottom=False,left=False,right=False,top=False)
            plt.box(False)
            plt.xlabel('posterior')
            plt.show()


        result = {
            'prob_volume': posterior_volume, # [BS,Hn,Wn,H,W]
            'sampled_normals': sampled_normals, # [BS,3,N,H,W]
            'normal_volume': normal_volume, # [BS,3,Hn,Wn,H,W]
            'likelihood_volume': likelihood_volume, # [BS,Hn,Wn,H,W]
            'prior_volume': prior_volume, # [BS,Hn,Wn,H,W]
            'pred_normal': pred_normal, # [BS,3,H,W]
        }

        return result

    # log_rmap: BS,3,Hn,Wn
    # hypotheses_normal: BS,3,N,H,W
    def compute_hypotheses_likelihood(self, log_img, log_rmap, hypotheses_normal):
        BS,C,N,H,W = hypotheses_normal.size()
        nx, ny = torch.unbind(hypotheses_normal, dim=1)[:2]
        hypotheses_mask = ((nx**2 + ny**2) <= 1.0).float()[:,None,:,:,:]
        hypotheses_mask *= (torch.sum(log_img[:,:,None,:,:], dim=1, keepdim=True) > 0.0).float()
        
        grid = torch.stack([nx, -ny], dim=-1).view(BS,N*H,W,2)
        sampled_log_radiance = F.grid_sample(log_rmap, grid, mode='nearest', padding_mode='zeros', align_corners=False)
        sampled_log_radiance = sampled_log_radiance.view(BS,3,N,H,W)
        hypotheses_mask *= (torch.sum(sampled_log_radiance, dim=1, keepdim=True) > 0.0).float()
        
        e = torch.sum(torch.abs(log_img[:,:,None,:,:] - sampled_log_radiance), dim=1, keepdim=True)
        hypotheses_likelihood = torch.exp(-F.softplus(self.likelihood_param) * e) # [BS,1,N,H,W]
        hypotheses_likelihood = hypotheses_likelihood * hypotheses_mask
        
        return hypotheses_likelihood
    

class RMFeaNet(nn.Module):
    def __init__(self, num_fea_in=4, num_fea_out=32):
        super(RMFeaNet, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.downsample = nn.AvgPool2d(2, stride=2)
        
        self.conv0 = ConvBnReLU(num_fea_in, 64, bias=True)

        self.conv1 = ConvBnReLU(64, 128, bias=True)
        self.conv2 = ConvBnReLU(128, 128, bias=True)

        self.conv3 = ConvBnReLU(128, 256, bias=True)
        self.conv4 = ConvBnReLU(256, 256, bias=True)

        self.conv5 = ConvBnReLU(256, 512, bias=True)
        self.conv6 = ConvBnReLU(512, 512, bias=True)

        self.conv_coarse_fea = nn.Sequential(
            ConvBnReLU(512, 512, bias=True),
            nn.Conv2d(512, num_fea_out, 3, 1, 1),
        )

        self.conv7 = ConvBnReLU(512, 256, bias=True)

        self.conv9 = ConvBnReLU(512, 128, bias=True)

        self.conv11 = ConvBnReLU(256, 64, bias=True)

        self.conv12 = nn.Conv2d(128, num_fea_out, 3, stride=1, padding=1)

        #self.conv_illum = nn.Sequential(
        #    nn.AvgPool2d(2, stride=2),
        #    ConvBnReLU(512, 512, bias=True),
        #    ConvBnReLU(512, 512, bias=True), # 512x8x8
        #    nn.AvgPool2d(2, stride=2),
        #    ConvBnReLU(512, 512, bias=True),
        #    ConvBnReLU(512, 512, bias=True),
        #    ConvBnReLU(512, 128, bias=True), # 128x4x4
        #)

        #self.linear = nn.Sequential(
        #    nn.Linear(128*4*4,512),
        #    nn.ReLU(),
        #    nn.Linear(512,512),
        #    nn.ReLU(),
        #    nn.Linear(512,512),
        #    nn.ReLU(),
        #    nn.Linear(512,256),
        #)
        
    def forward(self, x):
        conv0 = self.conv0(x[...,64:192,64:192]) # 64x128x128
        conv2 = self.conv2(self.conv1(self.downsample(conv0))) # 128x64x64
        conv4 = self.conv4(self.conv3(self.downsample(conv2))) # 256x32x32
        conv6 = self.conv6(self.conv5(self.downsample(conv4))) # 512x16x16
        coarse_fea_rmap = self.conv_coarse_fea(self.downsample(conv6)) # 16x8x8
        #return coarse_fea_rmap
        x = torch.cat([conv4, self.upsample(self.conv7(conv6))], dim=1)
        x = torch.cat([conv2, self.upsample(self.conv9(x))], dim=1)
        x = torch.cat([conv0, self.upsample(self.conv11(x))], dim=1)
        fine_fea_rmap = self.conv12(x)
        return coarse_fea_rmap, fine_fea_rmap 
        x = self.conv_illum(conv6).view(x.size(0),-1)
        illum_code = self.linear(x)
        return illum_code


class SimpleSfSNet(nn.Module):
    def __init__(self, num_gaussians=64, wo_lambertian=False):
        super(SimpleSfSNet, self).__init__()

        self.wo_lambertian = wo_lambertian
        if not wo_lambertian:
            self.img_fea_net = UNet(3+3,32+32+32)
            self.rm_net = RMFeaNet(3+3+3,32)
        else:
            self.img_fea_net = UNet(3,32+32+32)
            self.rm_net = RMFeaNet(3+3,32)

        #self.local_fea_net = nn.Sequential(
        #    ConvBnReLU(3, 64, bias=True),
        #    ConvBnReLU(64, 64, bias=True),
        #    ConvBnReLU(64, 64, bias=True),
        #    ConvBnReLU(64, 64, bias=True),
        #    nn.Conv2d(64, 16, 3, stride=1, padding=1)
        #)

        #self.orientation_wise_net = nn.Sequential(
        #    ConvBnReLU3D(32+16, 64, 1, 1, 0, bias=True),
        #    ConvBnReLU3D(64, 64, 1, 1, 0, bias=True),
        #    ConvBnReLU3D(64, 64, 1, 1, 0, bias=True),
        #    ConvBnReLU3D(64, 64, 1, 1, 0, bias=True),
        #    nn.Conv3d(64, 1, 1, stride=1, padding=0)
        #)

        self.log_pixel_wise_kappa = nn.Parameter(torch.tensor([3.,], requires_grad=True))

        #self.pixel_wise_net = nn.Sequential(
        #    ConvBnReLU(64, 256, 1, 1, 0, bias=True),
        #    ConvBnReLU(256, 256, 1, 1, 0, bias=True),
        #    ConvBnReLU(256, 256, 1, 1, 0, bias=True),
        #    ConvBnReLU(256, 256, 1, 1, 0, bias=True),
        #    nn.Conv2d(256, 256, 1, stride=1, padding=0)
        #)

        self.agg_net = UNet(32+64,64)

        self.refine_net = nn.Sequential(
            ConvBnReLU(64+64, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            nn.Conv2d(256, 256, 1, stride=1, padding=0)
        )

        self.fine_net_conv = UNet(256+3+32+32, 32)

        self.fine_net = nn.Sequential(
            ConvBnReLU(32+3+32+32, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            nn.Conv2d(256, 2, 1, stride=1, padding=0)
        )

        self.decoder = nn.Sequential(
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            ConvBnReLU(256, 256, 1, 1, 0, bias=True),
            nn.Conv2d(256, 4*num_gaussians, 1, stride=1, padding=0)
        )

    # img:  (BS*3*H*W)
    # rmaps: list of (BS*3*H*W)
    def forward(self, img, diffuse_img, rmap, diffuse_rmap, num_itr=2, initial_normal=None):
        BS,C,Hn,Wn = rmap.size()
        H,W = img.size()[2:4]
        Hn,Wn = rmap.size()[2:4]
        device = img.device

        img = pad_invalid_values(img)
        rmap = pad_invalid_values(rmap)
        diffuse_img = pad_invalid_values(diffuse_img)
        diffuse_rmap = pad_invalid_values(diffuse_rmap)

        mask = (torch.max(img, dim=1, keepdim=True)[0] > 0.0).float()

        # make mask of rmap
        rmap_normal = torch.stack(create_normal_grid((Wn,Hn), projection_mode='probe').to(device).unbind(-1), dim=0)[None]
        rmap_mask = (rmap_normal[:,2:3] > 0.0).float()
        rmap_normal = (rmap_normal * rmap_mask).repeat(BS,1,1,1)

        # scale normalization & apply log function
        mean_color = torch.sum(img*mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
        log_img = torch.log1p(100.0 * img / mean_color[:,:,None,None]) * mask
        log_rmap = torch.log1p(100.0 * rmap / mean_color[:,:,None,None]) * rmap_mask

        mean_color = torch.sum(diffuse_img*mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
        log_diffuse_img = torch.log1p(100.0 * diffuse_img / mean_color[:,:,None,None]) * mask
        mean_color = torch.sum(diffuse_rmap*rmap_mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
        log_diffuse_rmap = torch.log1p(100.0 * diffuse_rmap / mean_color[:,:,None,None]) * rmap_mask

        if not self.wo_lambertian:
            log_img = torch.cat([log_img, log_diffuse_img], dim=1)
            log_rmap = torch.cat([log_rmap, log_diffuse_rmap], dim=1)

        # image feature extraction
        fea_img = self.img_fea_net(log_img) # BSx32xHxW
        fea_img_coarse = fea_img[:,:32]
        fea_img_context = fea_img[:,32:64]
        fea_img_fine = fea_img[:,64:96]
        #local_fea_img = self.local_fea_net(log_img)
        
        if min(H,W) > 128:
            reso = min(H,W)
            log_img_ = F.interpolate(
                log_img, 
                size=((H//reso)*128, (W//reso)*128), 
                mode='bilinear', 
                align_corners=False, 
                antialias=True
            )
            fea_img_ = self.img_fea_net(log_img_)
            fea_img_coarse_hireso = fea_img_coarse
            fea_img_coarse = fea_img_[:,:32]
            fea_img_context = fea_img_[:,32:64]

        # RM Feature Extraction
        if True:
            fea_rmap_coarse, fea_rmap_fine = self.rm_net(torch.cat([log_rmap, rmap_normal], dim=1)) # BSx16x8x8
        else:
            fea_rmap_coarse = F.interpolate(log_rmap,size=(8,8), mode='area')
            fea_rmap_fine = log_rmap

        # Pixel-Wise Estimation
        fea_rmap_ = fea_rmap_coarse[:,:,:,:,None,None].repeat(1,1,1,1,128,128).view(BS,-1,fea_rmap_coarse.size(2)*fea_rmap_coarse.size(3),128,128)
        if True:
            fea_img_ = fea_img_coarse[:,:,None,:,:].repeat(1,1,fea_rmap_.size(2),1,1)
        else:
            fea_img_ = log_img[:,:,None,:,:].repeat(1,1,fea_rmap_.size(2),1,1)
        #pixel_wise_sfs_fea = self.orientation_wise_net(
        #    torch.cat([
        #        fea_img_, 
        #        fea_rmap_
        #    ], dim=1)
        #)[:,0,:,:,:] * mask # BSx64xHxW
        pixel_wise_sfs_fea = -torch.sum((fea_img_ - fea_rmap_)**2, dim=1)

        rmap_normal = create_normal_grid(16,'probe')[4:12,4:12].to(device).view(1,64,3,1,1)

        hemi_mask = (rmap_normal[:,:,2] > 0.0).float()
        pixel_wise_sfs_fea = F.softmax(pixel_wise_sfs_fea, dim=1) * hemi_mask
        pixel_wise_sfs_fea = pixel_wise_sfs_fea / (torch.sum(pixel_wise_sfs_fea, dim=1, keepdim=True) + 1e-12)

        pixel_wise_pi = pixel_wise_sfs_fea[:,:,None] # BSx64x1xHxW
        pixel_wise_mu = rmap_normal.repeat(BS,1,1,128,128)
        pixel_wise_kappa = self.log_pixel_wise_kappa.exp()[None,:,None,None,None].repeat(BS,64,1,128,128)

        pixel_wise_est_normal = torch.sum(pixel_wise_pi * pixel_wise_mu, dim=1)
        pixel_wise_est_normal = pixel_wise_est_normal / torch.sqrt(torch.sum(pixel_wise_est_normal**2, dim=1, keepdim=True) + 1e-6)
        pixel_wise_est_normal = pixel_wise_est_normal * F.interpolate(mask, (128,128), mode='nearest')

        pixel_wise_sfs_result = (pixel_wise_pi, pixel_wise_kappa, pixel_wise_mu, pixel_wise_est_normal)

        if False:
            from .sfs_utils import plot_normal_map, plot_hdr
            import matplotlib.pyplot as plt
            plt.subplot(1,6,1)
            plot_hdr(img)
            plt.subplot(1,6,2)
            plot_hdr(rmap[...,64:192,64:192])
            plt.subplot(1,6,3)
            plot_hdr(img[...,64:65,64:65])
            plt.subplot(1,6,4)
            plt.imshow(pixel_wise_pi[0,:,0,64,64].view(8,8).detach().cpu())
            plt.subplot(1,6,5)
            plt.imshow(pixel_wise_kappa[0,:,0,64,64].view(8,8).detach().cpu())
            plt.subplot(1,6,6)
            plot_normal_map(pixel_wise_mu[:,:,:,64,64].transpose(-1,-2).contiguous().view(BS,3,8,8))
            plt.show()

        #pixel_wise_sfs_code = self.pixel_wise_net(pixel_wise_sfs_fea) # BSx256xHxW

        # Context Aggregation
        agg_fea_img = self.agg_net(
            torch.cat([
                fea_img_context,
                pixel_wise_sfs_fea
            ], dim=1)
        ) # BSx64xHxW

        if min(H,W) > 128:
            # Hireso Pixel-Wise Estimation
            fea_rmap_ = fea_rmap_coarse[:,:,:,:,None,None].repeat(1,1,1,1,H,W).view(BS,-1,fea_rmap_coarse.size(2)*fea_rmap_coarse.size(3),H,W)
            if True:
                fea_img_ = fea_img_coarse_hireso[:,:,None,:,:].repeat(1,1,fea_rmap_.size(2),1,1)
            else:
                fea_img_ = log_img[:,:,None,:,:].repeat(1,1,fea_rmap_.size(2),1,1)
            #pixel_wise_sfs_fea = self.orientation_wise_net(
            #    torch.cat([
            #        fea_img_, 
            #        fea_rmap_
            #    ], dim=1)
            #)[:,0,:,:,:] * mask # BSx64xHxW
            pixel_wise_sfs_fea = -torch.sum((fea_img_ - fea_rmap_)**2, dim=1)

            rmap_normal = create_normal_grid(16,'probe')[4:12,4:12].to(device).view(1,64,3,1,1)

            hemi_mask = (rmap_normal[:,:,2] > 0.0).float()
            pixel_wise_sfs_fea = F.softmax(pixel_wise_sfs_fea, dim=1) * hemi_mask
            pixel_wise_sfs_fea= pixel_wise_sfs_fea / (torch.sum(pixel_wise_sfs_fea, dim=1, keepdim=True) + 1e-12)

            agg_fea_img = F.interpolate(
                agg_fea_img, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False, 
                antialias=True
            )

        # Pixel-Wise Refinement
        final_sfs_code = self.refine_net(
            torch.cat([
                agg_fea_img, 
                pixel_wise_sfs_fea
            ], dim=1)
        ) # BSx256xHxW

        # decode sfs features
        def decode(sfs_fea):
            decoded = self.decoder(sfs_fea) # BSx4NxHxW
            N = decoded.size(1) // 4

            p = decoded[:,:N]
            q = decoded[:,N:2*N]
            kappa = torch.clamp(decoded[:,2*N:3*N],None,30).exp()
            pi = F.softmax(decoded[:,3*N:4*N], dim=1)

            l = torch.sqrt(p**2 + q**2 + 1.0)
            nx = p / l
            ny = q / l
            nz = 1.0 / l
            mu = torch.stack([nx,ny,nz], dim=2) # BSxNx3xHxW

            kappa = kappa[:,:,None,:,:] # BSxNx1xHxW
            pi = pi[:,:,None,:,:] # BSxNx1xHxW

            # compute expected surface normal orientation
            est_normal = torch.sum(pi * mu, dim=1)
            est_normal = est_normal / torch.sqrt(torch.sum(est_normal**2, dim=1, keepdim=True) + 1e-6)
            est_normal = est_normal * mask

            return pi, kappa, mu, est_normal

        #pixel_wise_sfs_result = decode(pixel_wise_sfs_code)
        final_sfs_result = decode(final_sfs_code)

        est_normal = final_sfs_result[-1]
        if not (initial_normal is None):
            est_normal = initial_normal

        normal_results = [est_normal, ]

        for _ in range(num_itr):
            sampled_fea_rmap_fine = sample_rmap(fea_rmap_fine, est_normal, projection_mode='hemi_probe', interep_mode='bilinear')

            fine_code = self.fine_net_conv(
                torch.cat([
                    final_sfs_code, 
                    est_normal, # 3
                    fea_img_fine, # 32
                    sampled_fea_rmap_fine, # 32
                ], dim=1)
            )

            p,q = self.fine_net(
                torch.cat([
                    fine_code, 
                    est_normal, # 3
                    fea_img_fine, # 32
                    sampled_fea_rmap_fine, # 32
                ], dim=1)
            ).unbind(1) # BSx256xHxW

            l = torch.sqrt(p**2 + q**2 + 1.0)
            nx = p / l
            ny = q / l
            nz = 1.0 / l
            est_normal = torch.stack([nx,ny,nz], dim=1) * mask # BSxNx3xHxW

            normal_results.append(est_normal)

            if False:#not (initial_normal is None):
                p0 = initial_normal[:,0] / torch.clamp(initial_normal[:,2], 1e-3, None)
                q0 = initial_normal[:,1] / torch.clamp(initial_normal[:,2], 1e-3, None)

                ones = torch.ones_like(p0.view(BS,-1))
                zeros = torch.zeros_like(p0.view(BS,-1))
                w = 1. / torch.sqrt(p0**2 + q0**2 + 1.).view(BS,-1,1)
                w = 1. / torch.sqrt(p**2 + q**2 + 1.).view(BS,-1,1)
                w = w * mask.view(BS,-1,1)
                a1 = w * torch.stack([p0.view(BS,-1), ones, zeros], dim=-1) # BS*(H*W)*3
                a2 = w * torch.stack([q0.view(BS,-1), zeros, ones], dim=-1) # BS*(H*W)*3

                b1 = w * p.view(BS,-1,1)
                b2 = w * q.view(BS,-1,1)
                A = torch.cat([a1,a2], dim=-2)
                b = torch.cat([b1,b2], dim=-2)

                params = torch.linalg.lstsq(A,b).solution # BS*3*1
                p0_ = torch.stack([p0.view(BS,-1), ones, zeros], dim=-1) @ params
                q0_ = torch.stack([q0.view(BS,-1), zeros, ones], dim=-1) @ params
                p0_ = p0_.view(BS,H,W)
                q0_ = q0_.view(BS,H,W)

                l = torch.sqrt(p0_**2 + q0_**2 + 1.0)
                nx = p0_ / l
                ny = q0_ / l
                nz = 1.0 / l
                est_normal = torch.stack([nx,ny,nz], dim=1) * mask # BSxNx3xHxW

                if True:
                    print(params)
                    from .sfs_utils import plot_normal_map
                    import matplotlib.pyplot as plt
                    plt.subplot(1,3,1)
                    plot_normal_map(initial_normal)
                    plt.subplot(1,3,2)
                    plot_normal_map(est_normal)
                    plt.subplot(1,3,3)
                    plot_normal_map(normal_results[-1])
                    plt.show()



        return {
            'pixel_wise_sfs_result': pixel_wise_sfs_result,
            'sfs_result': final_sfs_result,
            'normal_results': normal_results,
            'est_normal': est_normal,
        }

class InitialSfSNet(nn.Module):
    def __init__(self):
        super(InitialSfSNet, self).__init__()
        self.unet1 = UNet(3,32)
        self.unet2 = UNet(3+32,2)

    def forward(self, img):
        BS,C,H,W = img.size()
        device = img.device

        img = pad_invalid_values(img)

        mask = (torch.max(img, dim=1, keepdim=True)[0] > 0.0).float()

        # scale normalization & apply log function
        mean_color = torch.sum(img*mask, dim=(2,3)) / torch.sum(mask, dim=(2,3)) # [BS,3]
        log_img = torch.log1p(100.0 * img / mean_color[:,:,None,None]) * mask

        p,q = self.unet2(torch.cat([log_img, self.unet1(log_img)], dim=1)).unbind(1)

        l = torch.sqrt(p**2 + q**2 + 1.0)
        nx = p / l
        ny = q / l
        nz = 1.0 / l
        est_normal = torch.stack([nx,ny,nz], dim=1) * mask # BSxNx3xHxW

        return est_normal