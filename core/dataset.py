import os
import numpy as np
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

from .rmap_utils import sample_rmap

class PreProcessedDataset(Dataset):
    def __init__(self, path):
        self.files = sorted(glob.glob(path+'/*.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

class TwoViewRealImageDataset(Dataset):

    def __init__(self, dataset_path, object_id, **kwargs):
        data_dir=dataset_path+'/'+str(object_id).zfill(5)
        imgs = []
        masks = []
        normal_maps = []
        depth_maps = []
        proj_matrices = []
        intrinsics = []
        extrinsics = []
        img_names = ['view-'+str(i).zfill(2) for i in range(100) if os.path.exists(data_dir+'/view-'+str(i).zfill(2)+'_m.png')]
        for img_name in img_names[:2]:
            is_jpg=False
            if os.path.exists(data_dir+'/'+img_name+'.exr'):
                img = cv2.imread(data_dir+'/'+img_name+'.exr', -1).astype(np.float32)
            elif os.path.exists(data_dir+'/'+img_name+'.png'):
                img = (cv2.imread(data_dir+'/'+img_name+'.png', 1).astype(np.float32) / 255.)**2.2
            elif os.path.exists(data_dir+'/'+img_name+'.jpg'):
                img = (cv2.imread(data_dir+'/'+img_name+'.jpg', 1).astype(np.float32) / 255.)**2.2
                is_jpg=True
            else:
                print('error')
                exit()
            mask = cv2.imread(data_dir+'/'+img_name+'_m.png', 0).astype(np.float32) / 255.0

            if os.path.exists(data_dir+'/'+img_name+'_intrinsics.npy'):
                K = np.load(data_dir+'/'+img_name+'_intrinsics.npy')
            elif is_jpg:
                image_height, image_width = img.shape[:2]
                import subprocess
                proc = subprocess.run(['exiftool', '-FocalLength35efl', '-s3', '-n', data_dir+'/'+img_name+'.jpg'], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                f0 = float(proc.stdout.decode("utf8").split('\n')[0]) / 36 * max([image_width, image_height])
                K = np.array([
                    [f0, 0.0, image_width/2], 
                    [0.0, f0, image_height/2], 
                    [0.0, 0.0, 1.0]]
                )
                print('intrinsic params from exif:')
                print(K)
            else:
                K = np.array([
                    [4000., 0., img.shape[1] // 2],
                    [0., 4000., img.shape[0] // 2],
                    [0., 0., 1.]
                ])

            if os.path.exists(data_dir+'/'+img_name+'_extrinsics.npy'):
                T = np.load(data_dir+'/'+img_name+'_extrinsics.npy')
            else:
                T = np.array([
                    [1., 0., 0., 0.,],
                    [0., 1., 0., 0.,],
                    [0., 0., 1., 2 * K[0,0] / max(img.shape[:2])],
                    [0., 0., 0., 1.,],
                ])

            # crop
            v_in, u_in = np.where(mask > 0.9)
            umin, umax = np.min(u_in), np.max(u_in)
            vmin, vmax = np.min(v_in), np.max(v_in)
            u0 = max(0, int(umin - 0.05 * (umax - umin)))
            u1 = min(img.shape[1], int(umax + 0.05 * (umax - umin)))
            v0 = max(0, int(vmin - 0.05 * (vmax - vmin)))
            v1 = min(img.shape[0], int(vmax + 0.05 * (vmax - vmin)))
            img = img[v0:v1, u0:u1]
            mask = mask[v0:v1, u0:u1]
            K = np.array([
                [1., 0., -u0],
                [0., 1., -v0],
                [0., 0., 1.]
            ]) @ K

            # paddiong for square shape
            if img.shape[0] > img.shape[1]:
                left = (img.shape[0] - img.shape[1]) // 2
                right = (img.shape[0] - img.shape[1] - left)
                img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_REFLECT, (0,0,0))
                mask = cv2.copyMakeBorder(mask, 0, 0, left, right, cv2.BORDER_CONSTANT, 0)
                K[0,2] += left
            elif img.shape[1] > img.shape[0]:
                top = (img.shape[1] - img.shape[0]) // 2
                bottom = (img.shape[1] - img.shape[0] - top)
                img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_REFLECT, (0,0,0))
                mask = cv2.copyMakeBorder(mask, top, bottom, 0, 0, cv2.BORDER_CONSTANT, 0)
                K[1,2] += top

            # resize
            fs = kwargs['img_size'][0] / img.shape[0]
            img = cv2.resize(img, tuple(kwargs['img_size']), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, tuple(kwargs['img_size']), interpolation=cv2.INTER_AREA)
            K = np.diag([fs,fs,1.]) @ K

            # binarize mask
            mask = (mask > 0.95).astype(np.float32)

            img = np.clip(img[...,::-1], 0, None)
            mask = np.clip(mask, 0, 1)

            imgs.append(torch.from_numpy(img.transpose((2,0,1))).float())
            masks.append(torch.from_numpy(mask[None,:,:]).float())

            # dummy ground truths
            intrinsics.append(torch.from_numpy(K).float())
            extrinsics.append(torch.from_numpy(T).float())
            P = np.eye(4)
            P[:3,:4] = K @ T[:3,:4]
            proj_matrices.append(torch.from_numpy(P).float())

            normal_maps.append(
                torch.from_numpy(np.stack([
                    0 * mask.astype(np.float32),
                    0 * mask.astype(np.float32),
                    mask.astype(np.float32),
                ], axis=0))
            )

            depth_maps.append(
                torch.from_numpy(T[2,3] * mask.astype(np.float32)[None,:,:])
            )


        self.imgs = torch.stack(imgs, dim=0)
        self.masks = torch.stack(masks, dim=0)
        self.intrinsics = torch.stack(intrinsics, dim=0)
        self.extrinsics = torch.stack(extrinsics, dim=0)
        self.proj_matrices = torch.stack(proj_matrices, dim=0)
        self.depth_maps = torch.stack(depth_maps, dim=0)
        self.normal_maps = torch.stack(normal_maps, dim=0)


        if False:
            print(self.intrinsics.size())

            import matplotlib.pyplot as plt
            from .sfs_utils import plot_hdr
            plt.subplot(2,1,1)
            plot_hdr(torch.cat(self.imgs.unbind(0), dim=-1))
            plt.subplot(2,1,2)
            plt.imshow(torch.cat(self.masks.unbind(0), dim=-1)[0])
            plt.show()


    def __len__(self):
        return 2

    def __getitem__(self, idx):
        view_indices = torch.tensor([idx,])


        return {
            'hdr_images': (self.imgs * self.masks)[idx:idx+1].clone(),
            'masks': self.masks[idx:idx+1].clone(),
            'gt_depths': self.depth_maps[idx:idx+1].clone(),
            'gt_normals': self.normal_maps[idx:idx+1].clone(),
            'intrinsics': self.intrinsics[idx:idx+1].clone(),
            'extrinsics': self.extrinsics[idx:idx+1].clone(),
            'proj_matrices': self.proj_matrices[idx:idx+1].clone(),
            #'depth_ranges': depth_ranges[idx:idx+1].clone(),
            'view_indices': view_indices[idx:idx+1],
        }

class nLMVSSynthDataset(Dataset):
    """Image to 3D dataset."""

    def __init__(self, dataset_path, **kwargs):
        """
        Args:
            img_dir (string): Path to the csv file with annotations.
            pcd_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_neighbors = 10
        self.use_crop = False
        self.img_size = None
        self.rmap_mode = 'polar'
        self.rmap_size = None
        self.mask_img = False
        self.use_illum = False
        self.use_diffuse_illum = False
        self.use_brdf_code = False
        self.use_diffuse_img = False
        self.use_diffuse_img_r05 = False
        self.use_diffuse_rmap = False
        self.normalize_world_scale = False
        self.use_soft_mask = False
        self.wo_global_illum = False
        options = kwargs
        if 'num_neighbors' in options:
            self.num_neighbors = options['num_neighbors']            
        if 'use_crop' in options:
            self.use_crop = options['use_crop']
        if 'img_size' in options:
            self.img_size = options['img_size']
        if 'rmap_mode' in options:
            self.rmap_mode = options['rmap_mode']
        if 'rmap_size' in options:
            self.rmap_size = options['rmap_size']
        if 'mask_img' in options:
            self.mask_img = options['mask_img']
        if 'use_illum' in options:
            self.use_illum = options['use_illum']
        if 'use_diffuse_illum' in options:
            self.use_diffuse_illum = options['use_diffuse_illum']
        if 'use_brdf_code' in options:
            self.use_brdf_code = options['use_brdf_code']
        if 'use_diffuse_img' in options:
            self.use_diffuse_img = options['use_diffuse_img']
        if 'use_diffuse_img_r05' in options:
            self.use_diffuse_img_r05 = options['use_diffuse_img_r05']
        if 'use_diffuse_rmap' in options:
            self.use_diffuse_rmap = options['use_diffuse_rmap']
        if 'normalize_world_scale' in options:
            self.normalize_world_scale = options['normalize_world_scale']
        if 'use_soft_mask' in options:
            self.use_soft_mask = options['use_soft_mask']
        if 'wo_global_illum' in options:
            self.wo_global_illum = options['wo_global_illum']

        if self.use_diffuse_img and self.wo_global_illum:
            self.use_diffuse_rmap = True

        self.instances = []
        for instance_dir in sorted(glob.glob(dataset_path+'/*')):
            idx_scan = int(instance_dir.split('/')[-1][4:])-1
            with open(instance_dir+'/pair.txt', 'r') as f:
                lines = f.readlines()
                num_views = int(lines[0])
                for idx_view in range(num_views):
                    ref_id = int(lines[1+2*idx_view])
                    src_ids = [int(s) for s in lines[2+2*idx_view].split()[1::2]][:self.num_neighbors]
                    ids = [ref_id] + src_ids
                    self.instances.append({
                        'instance_dir' : instance_dir,
                        'ids' : ids,
                        'idx_scan' : idx_scan
                    })


    def pad_invalid_values(self, x):
        mask = (torch.isnan(x) == False)
        mask *= (torch.isinf(x) == False)
        x[mask==False] = 0.0
        return torch.clamp(x, 0.0, 1e24)            

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance_dir = self.instances[idx]['instance_dir']
        idx_scan = self.instances[idx]['idx_scan']
        view_indices = torch.tensor(self.instances[idx]['ids'])

        if self.use_diffuse_img:
            diffuse_reflectance = torch.from_numpy(np.loadtxt(instance_dir+'/diffuse_reflectance.txt')).float()

        hdrs = []
        diffuse_hdrs = []
        diffuse_hdrs_r05 = []
        hdr_rmaps = []
        hdr_diffuse_rmaps = []
        depths = []
        normals = []
        extrinsics = []
        intrinsics = []
        depth_ranges = []
        for id_view in self.instances[idx]['ids']:
            depth = cv2.imread(instance_dir+'/Depths/'+str(id_view).zfill(8)+'.pfm', -1)[:,:,None]
            depth = np.transpose(depth, (2,0,1))
            depths.append(depth) 
            
            hdr = cv2.imread(instance_dir+'/images/'+str(id_view).zfill(8)+'.exr', -1)[:,:,::-1]
            hdr = np.transpose(hdr, (2,0,1))
            hdrs.append(hdr)

            hdr_rmap = cv2.imread(instance_dir+'/reflectance_maps/'+str(id_view).zfill(8)+'.exr', -1)[:,:,::-1]
            hdr_rmap = np.transpose(hdr_rmap, (2,0,1))
            hdr_rmaps.append(hdr_rmap)

            if self.use_diffuse_img:
                diffuse_hdr = cv2.imread(instance_dir+'/diffuse_images/'+str(id_view).zfill(8)+'.exr', -1)[:,:,::-1]
                diffuse_hdr = np.transpose(diffuse_hdr, (2,0,1))
                diffuse_hdrs.append(diffuse_hdr)

            if self.use_diffuse_img_r05:
                diffuse_hdr_r05 = cv2.imread(instance_dir+'/diffuse_images_r0.5/'+str(id_view).zfill(8)+'.exr', -1)[:,:,::-1]
                diffuse_hdr_r05 = np.transpose(diffuse_hdr_r05, (2,0,1))
                diffuse_hdrs_r05.append(diffuse_hdr_r05)

            if self.use_diffuse_rmap:
                hdr_diffuse_rmap = cv2.imread(instance_dir+'/diffuse_reflectance_maps/'+str(id_view).zfill(8)+'.exr', -1)[:,:,::-1]
                hdr_diffuse_rmap = np.transpose(hdr_diffuse_rmap, (2,0,1))
                hdr_diffuse_rmaps.append(hdr_diffuse_rmap)

            normal = cv2.imread(instance_dir+'/Normals/'+str(id_view).zfill(8)+'.exr', -1)[:,:,::-1]
            normal = 2 * normal - 1.0
            normal = np.transpose(normal, (2,0,1))
            normals.append(normal)                    

            with open(instance_dir+'/cams/'+str(id_view).zfill(8)+'_cam.txt', 'r') as f:
                lines = f.readlines()
            T = []
            for l in lines[1:5]:
                T.append([float(s) for s in l.split()])
            T  = np.array(T)
            extrinsics.append(T)
            K = []
            for l in lines[7:10]:
                K.append([float(s) for s in l.split()])
            K  = np.array(K)
            intrinsics.append(K)

            depth_min, depth_interval = [float(s) for s in lines[11].split()]
            depth_max = depth_min + 1.06 * depth_interval * 191
            depth_ranges.append([depth_min, depth_max])

        hdrs = torch.from_numpy(np.array(hdrs, dtype='float32'))
        hdr_rmaps = torch.from_numpy(np.array(hdr_rmaps, dtype='float32'))
        if self.use_diffuse_img:
            diffuse_hdrs = torch.from_numpy(np.array(diffuse_hdrs, dtype='float32'))
        if self.use_diffuse_img_r05:
            diffuse_hdrs_r05 = torch.from_numpy(np.array(diffuse_hdrs_r05, dtype='float32'))
        if self.use_diffuse_rmap:
            hdr_diffuse_rmaps = torch.from_numpy(np.array(hdr_diffuse_rmaps, dtype='float32'))
        depths = torch.from_numpy(np.array(depths, dtype='float32')) 
        normals = torch.from_numpy(np.array(normals, dtype='float32'))        
        extrinsics = torch.from_numpy(np.array(extrinsics, dtype='float32'))   
        intrinsics = torch.from_numpy(np.array(intrinsics, dtype='float32'))  
        depth_ranges = torch.from_numpy(np.array(depth_ranges, dtype='float32'))

        masks = torch.sum(normals**2, dim=1, keepdim=True)
        masks[masks < 1e-3] = 0.0
        masks[masks >= 1e-3] = 1.0
        masks[:,:,1:-1,:] *= masks[:,:,:-2,:] * masks[:,:,2:,:]
        masks[:,:,:,1:-1] *= masks[:,:,:,:-2] * masks[:,:,:,2:]  

        rmap_masks = (torch.sum(hdr_rmaps, dim=1, keepdim=True) > 0.0).float()

        if self.mask_img:
            hdrs *= masks
            if self.use_diffuse_img:
                diffuse_hdrs *= masks
            if self.use_diffuse_img_r05:
                diffuse_hdrs_r05 *= masks

        if self.rmap_mode in ['sphere', 'stereographic', 'probe']:
            # reflectance map as sphere
            Hn,Wn = hdr_rmaps.size()[-2:]
            v,u = torch.meshgrid(torch.arange(Hn), torch.arange(Wn))
            if self.rmap_mode == 'sphere':
                x = 2 * (u + 0.5) / Wn - 1
                y = -(2 * (v + 0.5) / Hn - 1)
                z = torch.sqrt(torch.clamp(1-x**2-y**2,0,None))
            elif self.rmap_mode == 'stereographic':
                p = 2.0 * (2 * (u + 0.5) / Wn - 1)
                q = 2.0 * (-(2 * (v + 0.5) / Hn - 1))
                r = 1 + p**2 + q**2

                x = 2 * p / r
                y = 2 * q / r
                z = -(-1 + p**2 + q**2) / r
            elif self.rmap_mode == 'probe':
                s = 2 * ((u + 0.5) / Wn - 0.5)
                t = -2 * ((v + 0.5) / Hn - 0.5)
                r = torch.clamp(torch.sqrt(s**2 + t**2),0,1)
                phi = np.pi * r

                x = s / r * torch.sin(phi)
                y = t / r * torch.sin(phi)
                z = torch.cos(phi)

            normal_vis = torch.stack([x,y,z], dim=0).to(hdr_rmaps.device)
            mask_vis = (z > 0.0).float()[None].to(hdr_rmaps.device)
            normal_vis *= mask_vis            
            # sample rmap
            hdr_rmaps_sphere = []
            hdr_diffuse_rmaps_sphere = []
            for idx_view in range(len(hdr_rmaps)):
                invR = torch.inverse(extrinsics[idx_view,:3,:3])
            
                global_normal_vis = invR[:,0][:,None,None] * normal_vis[0:1]
                for i in range(1,3):
                    global_normal_vis += -invR[:,i][:,None,None] * normal_vis[i:i+1]

        
                theta = torch.acos(torch.clamp(global_normal_vis[1], min=-1.0, max=1.0))[None]
                phi = torch.atan2(global_normal_vis[2], global_normal_vis[0])[None]

                phi[phi<0.0] += 2.0 * np.pi

                v = theta / np.pi * 2.0 - 1.0  # [1,H,W]
                u = phi / (2.0*np.pi) * 2.0 - 1.0  # [1,H,W]

                v, u = v.contiguous(), u.contiguous()
                grid = torch.stack([u,v], dim=3)# [1,H,W,2]

                hdr_rmap = F.grid_sample(hdr_rmaps[idx_view:idx_view+1], grid, mode='bilinear', padding_mode='border', align_corners=False)
                hdr_rmap *= mask_vis[None]
                hdr_rmaps_sphere.append(hdr_rmap)
                if self.use_diffuse_rmap:
                    hdr_diffuse_rmap = F.grid_sample(hdr_diffuse_rmaps[idx_view:idx_view+1], grid, mode='bilinear', padding_mode='border', align_corners=False)
                    hdr_diffuse_rmap *= mask_vis[None]
                    hdr_diffuse_rmaps_sphere.append(hdr_diffuse_rmap)

            rmap_masks[:,:,:,:] = mask_vis[None,None]
            hdr_rmaps = torch.cat(hdr_rmaps_sphere, dim=0) * rmap_masks
            if self.use_diffuse_rmap:
                hdr_diffuse_rmaps = torch.cat(hdr_diffuse_rmaps_sphere, dim=0) * rmap_masks

        if self.wo_global_illum:
            hdrs = sample_rmap(hdr_rmaps, normals, projection_mode=self.rmap_mode)

            if self.use_diffuse_img:
                diffuse_hdrs = sample_rmap(hdr_diffuse_rmaps, normals, projection_mode=self.rmap_mode)
        
        if self.use_crop:
            for i in range(len(depths)):
                v,u = torch.where(depths[i][0] > 0.0)
                vmin, vmax = torch.min(v), torch.max(v)
                umin, umax = torch.min(u), torch.max(u)
                screen_size = 1.05 * max((vmax-vmin), (umax-umin))
                object_area = torch.sum((depths[i][0] > 0.0).float())
                area_limit = 0.55
                if object_area > area_limit * (screen_size**2):
                    screen_size = np.sqrt(object_area / area_limit)
                u0 = (umin + umax - screen_size) / 2
                v0 = (vmin + vmax - screen_size) / 2
            
                v,u = torch.meshgrid(torch.arange(depths[i][0].size(0)), torch.arange(depths[i][0].size(1)))
                v = (v+0.5) / depths[i][0].size(0) * screen_size + v0
                u = (u+0.5) / depths[i][0].size(1) * screen_size + u0
                v = v / depths[i][0].size(0) * 2 - 1
                u = u / depths[i][0].size(1) * 2 - 1
                grid = torch.stack([u,v], dim=2)[None]
            
            
                hdrs[i:i+1] = F.grid_sample(hdrs[i:i+1], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                if self.use_diffuse_img:
                    diffuse_hdrs[i:i+1] = F.grid_sample(diffuse_hdrs[i:i+1], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                if self.use_diffuse_img_r05:
                    diffuse_hdrs_r05[i:i+1] = F.grid_sample(diffuse_hdrs_r05[i:i+1], grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                depths[i:i+1] = F.grid_sample(depths[i:i+1], grid, mode='nearest', padding_mode='zeros', align_corners=False)
                normals[i:i+1] = F.grid_sample(normals[i:i+1], grid, mode='nearest', padding_mode='zeros', align_corners=False)
                masks[i:i+1] = F.grid_sample(masks[i:i+1], grid, mode='nearest', padding_mode='zeros', align_corners=False)
            
                M = torch.tensor([
                    [depths[i][0].size(1)/screen_size, 0.0, -u0*depths[i][0].size(1)/screen_size],
                    [0.0, depths[i][0].size(0)/screen_size, -v0*depths[i][0].size(0)/screen_size],
                    [0.0, 0.0, 1.0]
                ])
                intrinsics[i] = torch.matmul(M, intrinsics[i])

        if self.img_size != None:
            # resize images
            prev_img_size = hdrs.size()[-2:]#[::-1]
        
            T = torch.tensor(
                [[self.img_size[1]/prev_img_size[1], 0.0, 0.0], [0.0, self.img_size[0]/prev_img_size[0], 0.0], [0.0, 0.0, 1.0]],
                device = intrinsics.device
            )
            intrinsics = torch.matmul(T[None], intrinsics)

            hdrs = F.interpolate(hdrs, self.img_size, mode='bilinear', antialias=True)
            if self.use_diffuse_img:
                diffuse_hdrs = F.interpolate(diffuse_hdrs, self.img_size, mode='bilinear', antialias=True)
            if self.use_diffuse_img_r05:
                diffuse_hdrs_r05 = F.interpolate(diffuse_hdrs_r05, self.img_size, mode='bilinear', antialias=True)
            depths = F.interpolate(depths, self.img_size, mode='nearest')            
            normals = F.interpolate(normals, self.img_size, mode='nearest')
            masks_nn = F.interpolate(masks, self.img_size, mode='nearest')
            masks = masks_nn if not self.use_soft_mask else F.interpolate(masks, self.img_size, mode='bilinear', antialias=True)
            #views = resize_tensors(views, img_size)

            depths *= masks_nn
            normals *= masks_nn

        if self.rmap_size != None:
            hdr_rmaps = F.interpolate(hdr_rmaps, self.rmap_size, mode='area')
            rmap_masks = F.interpolate(rmap_masks, self.rmap_size, mode='nearest')
            hdr_rmaps *= rmap_masks
            if self.use_diffuse_rmap:
                hdr_diffuse_rmaps = F.interpolate(hdr_diffuse_rmaps, self.rmap_size, mode='area')
                hdr_diffuse_rmaps *= rmap_masks

        world_scale = 1.0
        if self.normalize_world_scale:
            bbox_diagonal = 0.24039546982707555 * 0.6
            depths /= bbox_diagonal
            extrinsics[:,:3,3:4] /= bbox_diagonal
            world_scale = 1 / bbox_diagonal

        proj_matrices = []
        for K,T in zip(intrinsics, extrinsics):
            K_ = torch.cat([K, torch.tensor([0.0, 0.0, 0.0], device=K.device)[None,:]], dim=0)
            K_ = torch.cat([K_, torch.tensor([0.0, 0.0, 0.0, 1.0], device=K.device)[:,None]], dim=1)
            P = torch.matmul(K_,T)
            proj_matrices.append(P)
        proj_matrices = torch.stack(proj_matrices, dim=0).view(extrinsics.size())

        masks[torch.isnan(masks)] = 0
        masks[torch.isinf(masks)] = 1   
        if not self.use_soft_mask:
            masks[masks<0.5] = 0.0
            masks[masks>=0.5] = 1.0

        hdrs = self.pad_invalid_values(hdrs)
        if self.use_diffuse_img:
            diffuse_hdrs = self.pad_invalid_values(diffuse_hdrs)
        if self.use_diffuse_img_r05:
            diffuse_hdrs_r05 = self.pad_invalid_values(diffuse_hdrs_r05)
        hdr_rmaps = self.pad_invalid_values(hdr_rmaps)

        sample = {
            'hdr_images': hdrs, 
            'masks': masks,
            'hdr_rmaps': hdr_rmaps, 
            'rmap_masks': rmap_masks,
            'gt_depths': depths,
            'gt_normals': normals,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'proj_matrices': proj_matrices,
            'depth_ranges': depth_ranges,
            'idx_scan': idx_scan,
            'view_indices': view_indices,
            'world_scale' : world_scale,
        }

        if self.use_illum:
            envmap = cv2.imread(instance_dir+'/illumination.hdr', -1)[:,:,::-1]
            envmap = np.transpose(envmap, (2,0,1))
            envmap = torch.from_numpy(envmap.astype(np.float32))
            envmap = self.pad_invalid_values(envmap)
            sample['illum_map'] = envmap
        if self.use_diffuse_illum:
            envmap = cv2.imread(instance_dir+'/diffuse_illumination.hdr', -1)[:,:,::-1]
            envmap = np.transpose(envmap, (2,0,1))
            envmap = torch.from_numpy(envmap.astype(np.float32))
            envmap = self.pad_invalid_values(envmap)
            sample['diffuse_illum_map'] = envmap
        if self.use_brdf_code:
            d = torch.load(instance_dir+'/brdf_code.pt')
            sample['embed_code'] = d['embed_code'].float()
            sample['log_color'] = d['log_color'].float()
        if self.use_diffuse_img:
            sample['hdr_diffuse_images'] = diffuse_hdrs
            sample['diffuse_reflectance'] = diffuse_reflectance
        if self.use_diffuse_img_r05:
            sample['hdr_diffuse_images_r05'] = diffuse_hdrs_r05
        if self.use_diffuse_rmap:
            hdr_diffuse_rmaps = self.pad_invalid_values(hdr_diffuse_rmaps)
            sample['hdr_diffuse_rmaps'] = hdr_diffuse_rmaps

        #if self.transform:
        #    sample = self.transform(sample)

        return sample

class DrexelNatGeomDataset(Dataset):
    def __init__(self, dataset_path, options=None):
        self.img_reso = 128
        self.instances = []
        for illum_dir in sorted(glob.glob(dataset_path+'/*')):
            if not os.path.isdir(illum_dir):
                continue
            illum_path = glob.glob(illum_dir+'/*.exr')[0]
            for instance_dir in sorted(glob.glob(illum_dir+'/*')):
                if not os.path.isdir(instance_dir):
                    continue
                img_full_path = glob.glob(instance_dir+'/*_full.exr')[0]
                mask_full_path = glob.glob(instance_dir+'/*_m_full.png')[0]
                normal_full_path = glob.glob(instance_dir+'/*_n_full.npy')[0]
                #img_small_path = glob.glob(instance_dir+'/*_small.exr')[0]
                #mask_small_path = glob.glob(instance_dir+'/*_m_small.png')[0]
                #normal_small_path = glob.glob(instance_dir+'/*_n_small.npy')[0]
                self.instances.append({
                    'illum_path': illum_path,
                    'img_full_path': img_full_path,
                    'mask_full_path': mask_full_path,
                    'normal_full_path': normal_full_path,
                    #'img_small_path': img_small_path,
                    #'mask_small_path': mask_small_path,
                    #'normal_small_path': normal_small_path,
                })

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        
        illum = np.clip(cv2.imread(instance['illum_path'], -1)[:,:,::-1], 0, None)
        img = np.clip(cv2.imread(instance['img_full_path'], -1)[:,:,::-1], 0,None)
        mask = cv2.imread(instance['mask_full_path'], 0).astype(np.float) / 255.0
        normal = np.load(instance['normal_full_path'])
        
        # adjust dynamic range
        #s = np.exp(np.sum(np.log(img+1e-5)*mask[:,:,None] / (3*np.sum(mask))))
        #illum *= 0.05 / s
        #img *= 0.05 / s
        
        # to torch tensor
        illum = torch.from_numpy(illum.transpose(2,0,1).astype(np.float32))
        img = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))
        mask = torch.from_numpy(mask[None,:,:].astype(np.float32))
        normal = torch.from_numpy(normal.transpose(2,0,1).astype(np.float32))
        
        # to square image
        H0,W0 = img.size()[1:3]
        img_reso = int(1.02 * max([H0,W0]))
        y,x = torch.meshgrid(torch.arange(img_reso), torch.arange(img_reso))
        x = (x + 0.5) - 0.5 * img_reso
        y = (y + 0.5) - 0.5 * img_reso
        x = x / (0.5 * W0)
        y = y / (0.5 * H0)
        grid = torch.stack([x,y], dim=2)[None]
        
        img = F.grid_sample(img[None], grid, mode='bilinear', padding_mode='border', align_corners=False)[0]
        mask = F.grid_sample(mask[None], grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
        normal = F.grid_sample(normal[None], grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0]
        normal = normal / torch.sqrt(torch.clamp(torch.sum(normal**2, dim=0, keepdim=True),1e-4,None))
        
        # mask image
        img *= mask

        # adjust dynamic range
        x = []
        for i in range(3):
            x.append(torch.log(img[i:i+1]+1e-5)[mask.bool()])
        x = torch.cat(x)
        n = len(x)
        mean = torch.mean(x)
        m3 = torch.mean((x-mean)**3)
        s2 = 1 / (n - 1) * torch.sum((x-mean)**2)
        s3 = s2**(1.5)
        skewness = n**2 / (n-1) / (n-2) * m3 / s3        
        log_ofs = -3.09083648 + (-0.23820405) * skewness - mean
        scale = torch.exp(log_ofs)
        illum *= scale
        img *= scale

        # resize
        size = (self.img_reso, self.img_reso)
        img = F.interpolate(img[None], size, mode='area')[0]
        mask = F.interpolate(mask[None], size, mode='nearest')[0]
        normal = F.interpolate(normal[None], size, mode='nearest')[0]

        return {
            'illum': illum,
            'img': img,
            'mask': mask,
            'normal': normal
        }

class DrexelMultiNatGeomDataset(Dataset):
    def __init__(self, dataset_path, illumination_name, object_name, **kwargs):
        if os.path.exists(dataset_path+'/2D'):
            data_dir = dataset_path+'/2D/'+str(illumination_name)
        else:
            data_dir = dataset_path+'/'+str(illumination_name)
        self.img_size = (128,128)
        self.use_soft_mask = False
        if 'img_size' in kwargs:
            self.img_size = kwargs['img_size']
        if 'use_soft_mask' in kwargs:
            self.use_soft_mask = kwargs['use_soft_mask']

        H0, W0 = (5208, 3476)
        theta0 = 5
        sigma1 = 1
        sigma2 = 10
        num_neighbors = 4
        radiance_scale_factor = 0.0001

        # load illumination map
        if os.path.exists(data_dir+'/'+object_name+'/illumination.exr'):
            illumination_map = np.clip(cv2.imread(data_dir+'/'+object_name+'/illumination.exr', -1)[:,:,::-1], 0, None)
        else:
            illumination_map = np.clip(cv2.imread(data_dir+'/illumination.exr', -1)[:,:,::-1], 0, None)
        radiance_scale_factor = 1.0 / (2 * np.mean(illumination_map))
        illumination_map *= radiance_scale_factor
        self.illumination_map = torch.from_numpy(illumination_map.transpose(2,0,1).astype(np.float32)) 

        # load views
        with open(data_dir+'/'+object_name+'/views.txt', 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines if l[0] != '#']
        original_intrinsic = torch.tensor([float(s) for s in lines[0].split(" ")]).float().view(3,3)
        headers = []
        extrinsics = []
        offsets = []
        for l in lines[1:]:
            l = l.split(' ')
            headers.append(l[0][:-8])
            tvec = np.array([float(s) for s in l[1:4]])
            rvec = np.array([float(s) for s in l[4:7]])
            rot_matrix = cv2.Rodrigues(rvec)[0]
            T = np.append(rot_matrix, tvec[:,None], axis=1)
            T = np.append(T, np.array([[0.0,0.0,0.0,1.0]]), axis=0)
            T = torch.from_numpy(T).float()
            extrinsics.append(T)
            offset = torch.tensor([float(s) for s in l[7:9]]).float() # [u,v]
            offsets.append(offset)

        images = []
        masks = []
        depths = []
        normals = []
        intrinsics = []
        src_indices_list = []
        depth_ranges = []
        for idx_view, header in enumerate(headers):
            img = radiance_scale_factor * np.clip(cv2.imread(data_dir+'/'+object_name+'/'+header+'.exr', -1)[:,:,::-1], 0, None)
            img = torch.from_numpy(img.transpose(2,0,1).astype(np.float32))
            mask = cv2.imread(data_dir+'/'+object_name+'/'+header+'_m.png', 0).astype(np.float) / 255.0
            mask = torch.from_numpy(mask[None].astype(np.float32))
            depth = torch.from_numpy(np.load(data_dir+'/'+object_name+'/'+header+'_d.npy')[None].astype(np.float32))
            normal = torch.from_numpy(np.load(data_dir+'/'+object_name+'/'+header+'_n.npy').transpose(2,0,1).astype(np.float32))

            # binarize mask
            mask = (mask > 0.5).float()

            # compute intrinsic
            offset = offsets[idx_view]
            M = torch.tensor([
                [1.0, 0.0, -offset[0]],
                [0.0, 1.0, -offset[1]],
                [0.0, 0.0, 1.0]
            ])
            intrinsic = torch.matmul(M, original_intrinsic)


            # to square image
            if True:
                H1,W1 = img.size()[1:3]
                screen_size = int(1.05 * max([H1,W1]))
                object_area = torch.sum((depth[0] > 0.0).float())
                area_limit = 0.55
                if object_area > area_limit * (screen_size**2):
                    screen_size = np.sqrt(object_area / area_limit)
                y,x = torch.meshgrid(torch.arange(screen_size), torch.arange(screen_size))
                x = (x + 0.5) - 0.5 * screen_size
                y = (y + 0.5) - 0.5 * screen_size
                x = x / (0.5 * W1)
                y = y / (0.5 * H1)
                grid = torch.stack([x,y], dim=2)[None]

                img = F.grid_sample(img[None], grid, mode='bilinear', padding_mode='border', align_corners=False)[0]
                mask = F.grid_sample(mask[None], grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
                depth = F.grid_sample(depth[None], grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]
                normal = F.grid_sample(normal[None], grid, mode='nearest', padding_mode='zeros', align_corners=False)[0]

                # modify intrinsic
                M = torch.tensor([
                    [1.0, 0.0, 0.5*(screen_size - W1)],
                    [0.0, 1.0, 0.5*(screen_size - H1)],
                    [0.0, 0.0, 1.0]
                ])
                intrinsic = torch.matmul(M, intrinsic)

            if True:
                # mask image
                img = img * mask
                depth = depth * mask
                normal = normal * mask

            if not (self.img_size is None):
                # resize
                size = (self.img_size[0], self.img_size[1])
                img = F.interpolate(img[None], size, mode='bilinear', antialias=True)[0]
                mask_nn = F.interpolate(mask[None], size, mode='nearest')[0]
                mask = mask_nn if not self.use_soft_mask else F.interpolate(mask[None], size, mode='bilinear', antialias=True)[0]
                depth = F.interpolate(depth[None], size, mode='nearest')[0]
                normal = F.interpolate(normal[None], size, mode='nearest')[0]

                # modify intrinsics
                M = torch.tensor([
                    [self.img_size[1] / screen_size, 0.0, 0.0],
                    [0.0, self.img_size[0] / screen_size, 0.0],
                    [0.0, 0.0, 1.0]
                ])
                intrinsic = torch.matmul(M, intrinsic)

            # view selection
            T_ref = extrinsics[idx_view]
            view_ref = -torch.tensor([offset[0]+0.5*W1, offset[1]+0.5*H1, 1.0]).float()
            view_ref = torch.matmul(torch.inverse(original_intrinsic), view_ref[:,None])[:,0]
            view_ref = torch.matmul(torch.inverse(T_ref[:3,:3]), view_ref[:,None])[:,0]
            view_ref = view_ref / torch.linalg.norm(view_ref)
            l = []
            for idx_src, T_src in enumerate(extrinsics):
                if idx_src == idx_view:
                    continue
                ofs_src = offsets[idx_src]
                view_src = -torch.tensor([ofs_src[0]+0.5*W1, ofs_src[1]+0.5*H1, 1.0]).float()
                view_src = torch.matmul(torch.inverse(original_intrinsic), view_src[:,None])[:,0]
                view_src = torch.matmul(torch.inverse(T_src[:3,:3]), view_src[:,None])[:,0]
                view_src = view_src / torch.linalg.norm(view_src)
                baseline_angle_deg = torch.acos(torch.dot(view_ref, view_src)).item() / np.pi * 180
                if baseline_angle_deg <= theta0:
                    score = np.exp(-(baseline_angle_deg - theta0)**2 / (2*sigma1**2))
                else:
                    score = np.exp(-(baseline_angle_deg - theta0)**2 / (2*sigma2**2))
                l.append({
                    'idx_src' : idx_src, 
                    'score' : score, 
                    'baseline_angle_deg' : baseline_angle_deg
                })
            l.sort(key=lambda x: -x['score'])
            src_indices = [item['idx_src'] for item in l[:num_neighbors]]

            # depth ranges
            depth_mean = torch.linalg.norm(extrinsics[idx_view][:3,3])
            depth_min = depth_mean - 0.11
            depth_max = depth_mean + 0.11
            depth_range = torch.stack([depth_min, depth_max], dim=0)

            images.append(img)
            masks.append(mask)
            depths.append(depth)
            normals.append(normal)
            intrinsics.append(intrinsic)
            src_indices_list.append(src_indices)
            depth_ranges.append(depth_range)

        self.images = torch.stack(images, dim=0)
        self.masks = torch.stack(masks, dim=0)
        self.depths = torch.stack(depths, dim=0)
        self.normals = torch.stack(normals, dim=0)
        self.intrinsics = torch.stack(intrinsics, dim=0)
        self.extrinsics = torch.stack(extrinsics, dim=0)
        self.depth_ranges = torch.stack(depth_ranges, dim=0)
        self.src_indices_list = src_indices_list

        # adjust dynamic range
        if False:
            x = []
            for i in range(3):
                x.append(torch.log(self.images[:,i:i+1]+1e-5)[self.masks.bool()])
            x = torch.cat(x)
            n = len(x)
            mean = torch.mean(x)
            m3 = torch.mean((x-mean)**3)
            s2 = 1 / (n - 1) * torch.sum((x-mean)**2)
            s3 = s2**(1.5)
            skewness = n**2 / (n-1) / (n-2) * m3 / s3        
            log_ofs = -3.09083648 + (-0.23820405) * skewness - mean
            scale = torch.exp(log_ofs)
            self.illumination_map *= scale
            self.images *= scale

        #s = torch.exp(torch.sum(torch.log(self.images+1e-5)*self.masks / (3*torch.sum(self.masks)))).item()
        #self.illumination_map *= 0.05 / s
        #self.images *= 0.05 / s

    def __len__(self):
        return len(self.images)
    
    def compute_proj_matrix(self, intrinsic, extrinsic):
        K = torch.cat([intrinsic, torch.zeros((3,1), dtype=intrinsic.dtype, device=intrinsic.device)], dim=1)
        K = torch.cat([K, torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=K.dtype, device=K.device)], dim=0)
        return torch.matmul(K,extrinsic)

    def __getitem__(self, idx):
        images = [self.images[idx]]
        masks = [self.masks[idx]]
        depths = [self.depths[idx]]
        normals = [self.normals[idx]]
        intrinsics = [self.intrinsics[idx]]
        extrinsics = [self.extrinsics[idx]]
        proj_matrices = [self.compute_proj_matrix(self.intrinsics[idx], self.extrinsics[idx])]
        depth_ranges = [self.depth_ranges[idx]]
        view_indices = [idx]
        # gather neighboring views
        for idx_src in self.src_indices_list[idx]:
            images.append(self.images[idx_src])
            masks.append(self.masks[idx_src])
            depths.append(self.depths[idx_src])
            normals.append(self.normals[idx_src])
            intrinsics.append(self.intrinsics[idx_src])
            extrinsics.append(self.extrinsics[idx_src])
            proj_matrices.append(self.compute_proj_matrix(self.intrinsics[idx_src], self.extrinsics[idx_src]))
            depth_ranges.append(self.depth_ranges[idx_src])
            view_indices.append(idx_src)

        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        depths = torch.stack(depths, dim=0)
        normals = torch.stack(normals, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        extrinsics = torch.stack(extrinsics, dim=0)
        proj_matrices = torch.stack(proj_matrices, dim=0)
        depth_ranges = torch.stack(depth_ranges, dim=0)
        view_indices = torch.tensor(view_indices)

        world_scale = 1.0
        world_offset = 0.0
        if True:
            bbox_center = torch.tensor([0., 0.11, 0.]).float()
            r = np.sqrt(3 * 0.025**2)
            bbox_diagonal = 0.24039546982707555 * 0.6
            depths /= bbox_diagonal
            extrinsics[:,:3,3:4] /= bbox_diagonal
            scale = 0.8 / r
            M = torch.tensor([
                [1. / scale, 0., 0., bbox_center[0]],
                [0., 1. / scale, 0., bbox_center[1]],
                [0., 0., 1. / scale, bbox_center[2]],
                [0., 0., 0., 1.],
            ]).float()
            proj_matrices = proj_matrices @ M
            extrinsics = extrinsics @ M
            world_scale = world_scale * scale.item()
            world_offset = world_offset + bbox_center


        return {
            'hdr_images': images.clone(),
            'masks': masks.clone(),
            'gt_depths': depths.clone(),
            'gt_normals': normals.clone(),
            'intrinsics': intrinsics.clone(),
            'extrinsics': extrinsics.clone(),
            'proj_matrices': proj_matrices.clone(),
            'illum_map': self.illumination_map.clone(),
            'depth_ranges': depth_ranges.clone(),
            'view_indices': view_indices,
            'world_scale': world_scale,
            'world_offset': world_offset,
        }
