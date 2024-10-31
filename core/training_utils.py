import torch

from .criterion import ImageLogL1Loss, ImageGradientLoss, VGG16PerceptualLoss

# loss functions
image_loss = ImageLogL1Loss()
image_grad_loss = ImageGradientLoss()
perceptual_loss = VGG16PerceptualLoss()

def compute_depth_error(pred_depth, gt_depth, mask, depth_values):
    depth_range = depth_values[:,-1] - depth_values[:,0]
    error_map = torch.abs(pred_depth-gt_depth) * mask / depth_range[:,None,None,None]
    return torch.sum(error_map) / torch.sum(mask)

def normal_loss(pred_normal, gt_normal, mask):
    cosine = torch.sum(pred_normal * gt_normal, dim=1)
    angle_error = torch.acos(torch.clamp(cosine, -0.9999, 0.9999)) * mask[:,0]
    return torch.sum(angle_error) / torch.sum(mask[:,0])
