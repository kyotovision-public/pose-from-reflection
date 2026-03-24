import torch

from .criterion import ImageLogL1Loss, ImageGradientLoss, VGG16PerceptualLoss

# loss functions
image_loss = ImageLogL1Loss()
image_grad_loss = ImageGradientLoss()
perceptual_loss = VGG16PerceptualLoss()

def eval_rm_results(rmap_results, targets, loss_dict, logger, global_step=0, log_header=None):

    gt_rmap = targets['gt_rmap']
    sampled_img = targets['sampled_img']
    gt_mask_ = targets['gt_mask_']

    device = gt_rmap.device
    image_loss.to(device)
    image_grad_loss.to(device)
    perceptual_loss.to(device)

    if log_header is None:
        log_header = ''
    else:
        log_header += '/'

    loss_l1 = 0
    loss_grad = 0
    loss_perceptual = 0
    loss_img = 0
    for out in rmap_results:
        est_rmap = out['rmap']
        #est_mask = out['est_mask']
        est_img = out['est_img']

        if torch.any(torch.isnan(est_rmap)):
            print('est_rmap is NaN!!!!')

        loss_l1 = loss_l1 + image_loss(est_rmap, gt_rmap) #compute_log_l1_loss(est_diffuse_illum_map, gt_diffuse_illum_map, diffuse_illum_map_mask)
        loss_grad = loss_grad + image_grad_loss(
            est_rmap, 
            gt_rmap
        )
        loss_perceptual = loss_perceptual + perceptual_loss(
            est_rmap,
            gt_rmap
        )
        loss_img = loss_img + image_loss(
            est_img,
            sampled_img,
            mask = gt_mask_,
            num_scales=1
        )

    loss = loss_l1 + 1e-1 * loss_grad + 1e0 * loss_img + 1e-3 * loss_perceptual

    def add_to_dict(key, value, global_step):
        if key in loss_dict:
            loss_dict[key].append(value)
        else:
            loss_dict[key] = [value,]

        logger.add_scalar(key, value, global_step)

    add_to_dict(log_header+'Total_RM_Loss', loss.item(), global_step)
    add_to_dict(log_header+'L1_Loss', loss_l1.item(), global_step)
    add_to_dict(log_header+'Grad_Loss', loss_grad.item(), global_step)
    add_to_dict(log_header+'Image_Loss', loss_img.item(), global_step)
    add_to_dict(log_header+'Perceptual_Loss', loss_perceptual.item(), global_step)

    return loss, loss_dict, logger

def compute_depth_error(pred_depth, gt_depth, mask, depth_values):
    depth_range = depth_values[:,-1] - depth_values[:,0]
    error_map = torch.abs(pred_depth-gt_depth) * mask / depth_range[:,None,None,None]
    return torch.sum(error_map) / torch.sum(mask)

def normal_loss(pred_normal, gt_normal, mask):
    cosine = torch.sum(pred_normal * gt_normal, dim=1)
    angle_error = torch.acos(torch.clamp(cosine, -0.9999, 0.9999)) * mask[:,0]
    return torch.sum(angle_error) / torch.sum(mask[:,0])
