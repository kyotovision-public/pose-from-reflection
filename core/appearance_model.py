import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .rmap_utils import create_normal_grid

import tinycudann as tcnn

class NeuralRadianceField(torch.nn.Module):
    def __init__(self, homogeneous=False, ref_nerf=False, tiny_network=False, use_encoding=False, view_independent=False, num_outout_dims=3, num_rmaps=0):
        super(NeuralRadianceField, self).__init__()
        self.homogeneous = homogeneous
        self.view_independent = view_independent
        self.ref_nerf = ref_nerf
        self.use_encoding = use_encoding
        if num_rmaps > 0:
            self.rmap_codes = nn.Parameter(torch.randn([num_rmaps, 16], requires_grad=True))
            self.blending_field = tcnn.NetworkWithInputEncoding(#tcnn.Network( # WithInputEncoding(
                3, 
                num_rmaps,
                {
                    "otype": "Frequency", #"otype": "OneBlob", 
                    "n_frequencies": 4,#"n_bins": 64
                }, 
                {
                    "otype": "CutlassMLP", 
                    "activation": "ReLU", 
                    "output_activation": "None", 
                    "n_neurons": 128, 
                    "n_hidden_layers": 4
                }
            )
        else:
            self.rmap_codes = None

        # radiance field
        self.encoding = tcnn.Encoding(
            3, 
            {
                "otype": "Frequency",
                "n_frequencies": 4,
            }
        )
        num_input_dims = (16 if (num_rmaps > 0) else 0)
        num_input_dims += 9 if not use_encoding else 30
        if tiny_network:
            self.radiance_field = tcnn.Network( # WithInputEncoding(
                num_input_dims, 
                num_outout_dims,
                #{
                #    "otype": "Frequency", #"otype": "OneBlob", 
                #    "n_frequencies": 4,#"n_bins": 64
                #}, 
                {
                    "otype": "FullyFusedMLP", 
                    "activation": "ReLU", 
                    "output_activation": "None", 
                    "n_neurons": 128, 
                    "n_hidden_layers": 5
                }
            )
        else:
            self.radiance_field = tcnn.Network( # WithInputEncoding(
                num_input_dims, 
                num_outout_dims,
                #{
                #    "otype": "Frequency", #"otype": "OneBlob", 
                #    "n_frequencies": 4,#"n_bins": 64
                #}, 
                {
                    "otype": "CutlassMLP", 
                    "activation": "ReLU", 
                    "output_activation": "None", 
                    "n_neurons": 512, 
                    "n_hidden_layers": 4
                }
            )

    def generate_rmaps(self, extrinsics=None, x=None, out_size = 256, projection_mode='stereographic', rmap_index=0):
        assert not (extrinsics is None)
        num_views = len(extrinsics)
        v = -extrinsics[:,2:3,:3].repeat(1,out_size**2,1)

        n = create_normal_grid(out_size, projection_mode=projection_mode).to(v.device)
        n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=n.device)
        n = (extrinsics[:,None,:3,:3].transpose(-1,-2) @ n.view(1,-1,3,1))[...,0]

        if x is None:
            x = torch.zeros_like(v, dtype=v.dtype, device=v.device)

        if not (self.rmap_codes is None):
            rmap_code = self.rmap_codes[rmap_index][None,None,:].repeat(len(extrinsics),out_size**2,1)
        else:
            rmap_code = None

        return self.forward(x, n, v, extrinsics, rmap_code=rmap_code).transpose(-1,-2).view(-1,3, out_size, out_size)

    # x: num_views*num_rays*3
    def query_blending_weight(self, x, n=None, v=None, extrinsics=None, chunk_size=65536):
        num_views, num_rays = x.size(0), x.size(1)
        mask = (torch.sum(x**2, dim=-1) < 100.0)
        if not (n is None):
            mask = mask * (torch.sum(n**2, dim=-1) > 0.5)            
        pixel_vals = torch.zeros_like(x[...,:1], device=x.device).repeat(1,1,len(self.rmap_codes)).view(num_views*num_rays,-1)
        x_in_ = x.view(num_views*num_rays,-1)[mask.view(-1)]
        pixel_vals_ = []
        for i in range((len(x_in_) - 1) // chunk_size + 1):
            x_in_chunk = x_in_[chunk_size*i:chunk_size*(i+1)]
            pixel_vals_.append(self.blending_field(x_in_chunk))
        pixel_vals_ = torch.cat(pixel_vals_, dim=0)
        pixel_vals[mask.view(-1)] = pixel_vals_.float()
        pixel_vals = pixel_vals.view(num_views, num_rays, -1)

        pixel_vals = F.softmax(pixel_vals, dim=-1)

        return pixel_vals * mask.float()[...,None]

    # n: num_views*num_rays*3
    def forward(self, x, n, v, extrinsics, chunk_size=65536, rmap_code=None):
        num_views, num_rays = x.size(0), x.size(1)

        mask = (torch.sum(n**2, dim=-1) > 0.5)

        # camera coord to world coord
        #n = n * torch.tensor([1,-1,-1], dtype=n.dtype, device=n.device)
        #n = torch.sum(n[...,:,None] * extrinsics[:,None,:3,:3], dim=-2)

        if self.ref_nerf:
            r = -v + 2 * torch.sum(v * n, dim=-1, keepdim=True) * n
            if self.use_encoding:
                v_freq = self.encoding(r.view(num_views*num_rays,3)).view(num_views,num_rays,-1)
            else:
                v_freq = r
        else:
            if self.use_encoding:
                v_freq = self.encoding(v.view(num_views*num_rays,3)).view(num_views,num_rays,-1)
            else:
                v_freq = v

        if self.homogeneous:
            x = x * 0.0

        if self.view_independent:
            n = 0 * n
            v_freq = 0 * v_freq

        x_in = torch.cat([x,n,v_freq], dim=-1)
        if not (self.rmap_codes is None):
            x_in = torch.cat([x_in, rmap_code], dim=-1)

        pixel_vals = torch.zeros_like(n, device=n.device).view(num_views*num_rays,-1)
        x_in_ = x_in.view(num_views*num_rays,-1)[mask.view(-1)]
        pixel_vals_ = []
        for i in range((len(x_in_) - 1) // chunk_size + 1):
            x_in_chunk = x_in_[chunk_size*i:chunk_size*(i+1)]
            pixel_vals_.append(
                torch.clamp(-2 + self.radiance_field(x_in_chunk), np.log(1e-6), 30).exp()
            )
        pixel_vals_ = torch.cat(pixel_vals_, dim=0)
        #pixel_vals_ = torch.exp(-2 + self.radiance_field(x_in_))
        pixel_vals[mask.view(-1)] = pixel_vals_.float()
        pixel_vals = pixel_vals.view(num_views, num_rays, 3)

        #pixel_masks = torch.exp(-20 * torch.clamp(-torch.sum(v * n, dim=-1, keepdim=True), 0, None))

        return pixel_vals# * pixel_masks