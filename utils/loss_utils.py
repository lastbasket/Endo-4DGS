#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torch.nn as nn
import cv2
import numpy as np
from torchmetrics.regression import PearsonCorrCoef
from utils.general_utils import build_rotation
# from pytorch3d.transforms import quaternion_to_matrix
KEY_OUTPUT = 'metric_depth'



def get_smallest_axis(rot, scaling, return_idx=False):
        """Returns the smallest axis of the Gaussians.

        Args:
            return_idx (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rotation_matrices = build_rotation(rot)
        # print(rotation_matrices)
        # print(rotation_matrices.shape)
        # print(scaling.shape)
        # print((scaling.min(dim=-1)[1]).shape)
        # print(scaling)
        smallest_axis_idx = scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        # print(smallest_axis)
        # print(smallest_axis.shape)
        # exit()
        
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
def confidence_loss(gt, pred, confidence, mask):
    mask = torch.logical_or(torch.isnan(pred), \
        ~mask.expand(-1, pred.shape[1], -1, -1))
    mask = ~torch.logical_or(mask, \
        torch.isnan(confidence).expand(-1, pred.shape[1], -1, -1))
    gt = gt[mask]
    pred = pred[mask]
    confidence = confidence[mask[:, 0:1]]
    if min(pred.shape)==0 or min(confidence.shape)==0:
        return 0
    loss = F.mse_loss(gt, pred, reduction='mean')
    # print(torch.isnan(loss))
    loss = loss / ((2*confidence**2).mean()) + torch.log(confidence).mean()
    return loss
    
    
def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    nan_mask = ~torch.isnan(network_output)
    if mask is not None:
        if mask.shape[1] != network_output.shape[1]:
            mask = mask.expand(-1, network_output.shape[1], -1, -1)
    if mask is not None:
        return (torch.abs((network_output[nan_mask] - gt[nan_mask]))*mask[nan_mask]).mean()
    else:
        return (torch.abs((network_output[nan_mask] - gt[nan_mask]))).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def edge_aware_loss_v2(rgb, dep, mask=None):
    """Computes the smoothness loss for a deparity image
    The color image is used for edge-aware smoothness
    """
    dep = dep.permute(0,2,3,1)
    rgb = rgb.permute(0,2,3,1)
    mask = mask.permute(0,2,3,1)
    
    
    mean_dep = dep.mean(1, True).mean(2, True)#行&列的均值
    dep = dep / (mean_dep + 1e-7)#归一化处理

    grad_dep_x = torch.abs(dep[:, :, :-1, :] - dep[:, :, 1:, :])#x轴梯度
    grad_dep_y = torch.abs(dep[:, :-1, :, :] - dep[:, 1:, :, :])#y轴梯度

    grad_rgb_x = torch.mean(torch.abs(rgb[:, :, :-1, :] - rgb[:, :, 1:, :]), 3, keepdim=True)
    grad_rgb_y = torch.mean(torch.abs(rgb[:, :-1, :, :] - rgb[:, 1:, :, :]), 3, keepdim=True)

    grad_dep_x *= torch.exp(-grad_rgb_x)
    grad_dep_y *= torch.exp(-grad_rgb_y)
    # mask=torch.ones_like(dep)

    if mask is not None:
        grad_dep_x+=mask[:,:,:-1,:]*grad_dep_x
        grad_dep_y+=mask[:,:-1,:,:]*grad_dep_y
        
    return grad_dep_x.mean() + grad_dep_y.mean()


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)
  
class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)

def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    # return reduction(image_loss, 2 * M)
    return image_loss

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    # return reduction(image_loss, M)
    return image_loss

class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total
    
def compute_scale_and_shift(prediction, target, mask):

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    # print(a_00, a_01, a_11)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    # print('b0 b1: ', b_0, b_1)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()
    # print(valid)
    # print("a01 * b0, a00 * b1: ", a_01[valid] * b_0[valid], a_00[valid] * b_1[valid])
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target):
        #preprocessing
        mask = target > 0

        #calcul
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi
    
# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

            input = input[mask]
            target = target[mask]

        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)

        # n, c, h, w = g.shape
        # norm = 1/(h*w)
        # Dg = norm * torch.sum(g**2) - (0.85/(norm**2)) * (torch.sum(g))**2

        Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        if not return_interpolated:
            return loss

        return loss, intr_input

def extract_key(prediction, key):
    if isinstance(prediction, dict):
        return prediction[key]
    return prediction

def mae_loss(img1, img2, mask):
    mae_map = -torch.sum(img1 * img2, dim=1, keepdims=True) + 1
    loss_map = torch.abs(mae_map * mask)
    loss = torch.mean(loss_map)
    return loss

def grad(x):
    # x.shape : n, c, h, w
    diff_x = x[..., 1:, 1:] - x[..., 1:, :-1]
    diff_y = x[..., 1:, 1:] - x[..., :-1, 1:]
    # z = torch.ones_like(diff_x)
    # normal = torch.cat([-diff_x, -diff_y, -z], dim=1)
    # normal = F.normalize(normal, dim=1)
    # cv2.imwrite('norm.png', (normal*255).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
    # print(diff_x.max(), diff_y.max())
    mag = diff_x**2 + diff_y**2
    # angle_ratio
    angle = torch.atan(diff_y / (diff_x + 1e-10))
    return mag, angle

def grad_mask(mask):
    return mask[..., 1:, 1:] & mask[..., 1:, :-1] & mask[..., :-1, 1:]

class GradL1Loss(nn.Module):
    """Gradient loss"""
    def __init__(self):
        super(GradL1Loss, self).__init__()
        self.name = 'GradL1'
        self.pc_loss = PearsonCorrCoef().cuda()

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        mask = mask[None]
        input = extract_key(input, KEY_OUTPUT)
        if input.shape[-1] != target.shape[-1] and interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)
            intr_input = input
        else:
            intr_input = input

        grad_gt = grad(target)
        grad_pred = grad(input)
        mask_g = grad_mask(mask)
        # print(grad_pred[0].shape)
        # print(grad_gt[0].shape)
        # print(grad_pred[1].shape)
        
        # print(mask_g.shape)
        # print(grad_pred[1][mask_g].shape)
        # print(grad_gt[0][mask_g].shape)
        # print(grad_gt[1][mask_g].shape)
        # print(grad_gt[0].max())
        # print(grad_pred[0][mask_g].max())
        # print(grad_gt[1][mask_g].max())
        # print(grad_pred[1][mask_g].max())
        
        mask_mag = ~torch.logical_or(torch.isnan(grad_pred[0]), \
            ~mask_g.expand(-1, grad_pred[0].shape[1], -1, -1))
        mask_mag = mask_mag*grad_pred[0]>0
        # mask_ang = ~torch.logical_or(torch.isnan(grad_pred[1]), \
        #     ~mask_g.expand(-1, grad_pred[1].shape[1], -1, -1))
        # cv2.imwrite('mag.png', (grad_gt[0]*mask_g/(grad_gt[0]*mask_g).max()*255).squeeze(0).permute(1,2,0).detach().cpu().numpy().astype(np.uint8))
        # exit()
        # loss = (1 - self.pc_loss(grad_pred[0][mask_g][:, None], grad_gt[0][mask_g][:, None]))
        # loss = loss + (1 - self.pc_loss(grad_pred[1][mask_g][:, None], grad_gt[1][mask_g][:, None]))
        # print(grad_pred[0][mask_mag])
        # print(grad_pred[1][mask_mag])
        # print(torch.sum(torch.isnan(grad_pred[0])))
        # print(torch.sum(torch.isnan(grad_pred[1])))
        # print(self.pc_loss(grad_pred[0][mask_mag][:, None], grad_gt[0][mask_mag][:, None]))
        if min(grad_pred[0][mask_mag][:, None].shape) == 0:
            loss = 0
        else:
            loss = (1 - self.pc_loss(grad_pred[0][mask_mag][:, None], grad_gt[0][mask_mag][:, None]))
        # if min(grad_pred[1][mask_ang][:, None].shape) == 0:
        #     loss = loss + 0
        # else:
        #     loss = loss + (1 - self.pc_loss(grad_pred[1][mask_ang][:, None], grad_gt[1][mask_ang][:, None]))
            
        # loss = nn.functional.l1_loss(grad_pred[0][mask_g], grad_gt[0][mask_g])
        # loss = loss + \
        #     nn.functional.l1_loss(grad_pred[1][mask_g], grad_gt[1][mask_g])
        if not return_interpolated:
            return loss
        return loss, intr_input