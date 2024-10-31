import torch
import torch.nn as nn

import nvdiffrast.torch as dr

class MeshRenderer:
    def __init__(self, device=None):
        self.glctx = dr.RasterizeGLContext(device=device)

    # verts: Nverts*3, dtype=torch.float32
    # faces: Nfaces*3, dtype=torch.int32
    # proj_matrices: Nview*4*4
    # rspp: root of number of samples per pixel
    def render(self, verts, vert_normals, faces, proj_matrices, extrinsics, shading_func, resolution=(128,128), rspp=4, z_near=1e-3, z_far=10, orthographic_camera=False, level_set_fn=None):
        num_views = len(proj_matrices)

        if rspp != 1:
            resolution = (rspp * resolution[0], rspp * resolution[1])
            S = torch.diag(torch.tensor([rspp,rspp,1.,1.])).to(proj_matrices.device)
            proj_matrices = S @ proj_matrices

        # compute viewing direction
        v, u = torch.meshgrid(torch.arange(resolution[0]), torch.arange(resolution[1]))
        m = torch.stack([u,v,torch.ones_like(u)], dim=-1).to(proj_matrices.device) + 0.5 # H*W*3
        pixel_coords = m[None].repeat(len(proj_matrices),1,1,1) # Nview*H*W*3

        inv_proj = torch.inverse(proj_matrices)[:,:3,:3]
        pixel_ray_dirs = torch.sum(inv_proj[:,None,None,:,:] * pixel_coords[...,None,:], dim=-1)
        pixel_ray_dirs = pixel_ray_dirs / torch.sqrt(torch.sum(pixel_ray_dirs**2, dim=-1, keepdim=True) + 1e-6)

        if orthographic_camera:
            pixel_ray_dirs[:] = extrinsics[:,None,None,2,:3]
            pass

        # rasterize vertex positions and normals
        verts_proj_ = torch.sum(proj_matrices[:,None,:4,:3] * verts[None,:,None,:], dim=-1) + proj_matrices[:,None,:4,3]
        verts_proj_ = verts_proj_ # contains (u*depth, v*depth, depth, 1) * w
        M = torch.tensor([
            [2 / resolution[1], 0, -1, 0],
            [0, 2 / resolution[0], -1, 0],
            [0, 0, (z_far+z_near) / (z_far - z_near), -2 * z_near * z_far / (z_far - z_near)],
            [0, 0, 1, 0],
        ], device=verts_proj_.device)
        verts_ndc = (M @ verts_proj_[...,:,None])[...,0]
        verts_ndc = verts_ndc / verts_ndc[...,3:4]
        rast = dr.rasterize(self.glctx, verts_ndc, faces, resolution)[0]
        pixel_hard_masks = (rast[...,-1:] != 0).float()

        vert_attrs = torch.cat([verts, vert_normals, torch.ones_like(verts[...,:1], device=verts.device)], dim=-1)
        pixel_attrs = dr.interpolate(vert_attrs[None], rast, faces)[0]

        pixel_surf_coords = pixel_attrs[:,:,:,:3]
        pixel_surf_normals = pixel_attrs[:,:,:,3:6]
        pixel_masks = pixel_attrs[:,:,:,6:7]

        # make sure that dot(v,n) > 0
        vDn = torch.clamp(-torch.sum(pixel_surf_normals * pixel_ray_dirs, dim=-1, keepdim=True), None, 0)
        pixel_surf_normals = pixel_surf_normals + vDn * pixel_ray_dirs
        pixel_surf_normals = pixel_surf_normals * (vDn > -0.1).float()

        pixel_surf_normals = pixel_surf_normals / torch.sqrt(torch.sum(pixel_surf_normals**2, dim=-1, keepdim=True) + 1e-9)

        if False:#not (level_set_fn is None):
            # refine surf_coords and surf_normals
            for _ in range(0):
                pixel_sdf_vals, pixel_sdf_grads = level_set_fn(pixel_surf_coords.contiguous().view(-1,3))
                pixel_sdf_vals = pixel_sdf_vals.view(num_views, resolution[0], resolution[1], 1)
                pixel_sdf_grads = pixel_sdf_grads.view(num_views, resolution[0], resolution[1], 3)
                pixel_sdf_normals = pixel_sdf_grads / torch.sqrt(torch.sum(pixel_sdf_grads**2, dim=-1, keepdim=True) + 1e-6)
                pixel_vDn = torch.sum(pixel_sdf_normals * pixel_ray_dirs, dim=-1, keepdim=True)
                pixel_surf_coords = pixel_surf_coords - pixel_sdf_vals * pixel_vDn * pixel_ray_dirs
            pixel_sdf_grads = level_set_fn(pixel_surf_coords.contiguous().view(-1,3))[1].view(num_views, resolution[0], resolution[1], 3)
            pixel_surf_normals = pixel_sdf_grads / torch.sqrt(torch.sum(pixel_sdf_grads**2, dim=-1, keepdim=True) + 1e-6)

            pixel_surf_coords = pixel_surf_coords * pixel_masks
            pixel_surf_normals = pixel_surf_normals * pixel_masks

        if shading_func is None:
            pixel_vals = pixel_surf_normals
        else:
            pixel_vals = shading_func(
                pixel_surf_coords.view(num_views,-1,3),
                pixel_surf_normals.view(num_views,-1,3),
                -pixel_ray_dirs.view(num_views,-1,3),
                extrinsics,
            ).view(num_views, resolution[0], resolution[1],-1)

        pixel_vals = dr.antialias(pixel_vals.contiguous(), rast, verts_ndc, faces)
        pixel_masks = dr.antialias(pixel_masks.contiguous(), rast, verts_ndc, faces)
        pixel_surf_normals = dr.antialias(pixel_surf_normals.contiguous(), rast, verts_ndc, faces)

        pixel_depths = (extrinsics[:,None,None,2:3,:3] @ pixel_surf_coords[...,None] + extrinsics[:,None,None,2:3,3:4])[...,0]
        pixel_depths = pixel_depths * pixel_hard_masks + z_far * (1 - pixel_hard_masks)

        pixel_surf_normals_local = (extrinsics[:,None,None,:3,:3] @ pixel_surf_normals[...,None])[...,0]
        pixel_surf_normals_local = pixel_surf_normals_local * torch.tensor([1,-1,-1], device=pixel_surf_normals.device)

        img = pixel_vals.transpose(-1,-2).transpose(-2,-3)
        depth_map = pixel_depths.transpose(-1,-2).transpose(-2,-3)
        normal_map = pixel_surf_normals_local.transpose(-1,-2).transpose(-2,-3)
        mask = pixel_masks.transpose(-1,-2).transpose(-2,-3)
        surf_pos_map = pixel_surf_coords.transpose(-1,-2).transpose(-2,-3)

        if rspp != 1:
            img = torch.nn.functional.interpolate(
                img, scale_factor=1/rspp,mode='bilinear', align_corners=False, antialias=True
            )
            depth_map = torch.nn.functional.interpolate(
                depth_map, scale_factor=1/rspp,mode='nearest'
            )
            normal_map = torch.nn.functional.interpolate(
                normal_map, scale_factor=1/rspp,mode='bilinear', align_corners=False, antialias=True
            )
            mask = torch.nn.functional.interpolate(
                mask, scale_factor=1/rspp,mode='bilinear', align_corners=False, antialias=True
            )
            surf_pos_map = torch.nn.functional.interpolate(
                surf_pos_map, scale_factor=1/rspp,mode='nearest'
            )

        normal_map = normal_map / torch.sqrt(torch.sum(normal_map**2, dim=1, keepdim=True) + 1e-9)

        return img, depth_map, normal_map, mask, surf_pos_map
