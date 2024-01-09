"""
Differentiable volumetric implementation used by pi-GAN generator.
"""

import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import logging

from .math_utils_torch import *

def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""
    # rgb_sigma.shape: [batch_size, num_rays, num_steps, 4]
    # z_vals.shape: [batch_size, num_rays, num_steps, 1]

    # rgb_sigma 由siren网络训练得到
    rgbs = rgb_sigma[..., :3]   # rgb [batch_size, num_rays, num_steps, 3]
    sigmas = rgb_sigma[..., 3:] # sigma [batch_size, num_rays, num_steps, 1]

    # deltas 两个采样点之间的距离 d
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1] # 每两个采样点之间的距离
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1]) # 远平面 无穷远处
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std # 随机噪声 [batch_size, num_rays, num_steps, 1]

    # 计算 alpha
    if clamp_mode == 'softplus':
        alphas = 1-torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    
    # 计算 NeRF 中的 transmittance weights = aplhas * T_i 体渲染公式
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1] # [batch_size, num_rays, num_steps, 1]
    weights_sum = weights.sum(2) # [batch_size, num_rays, 1] 每条射线的权重和

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2) # [batch_size, num_rays, 3] 最终预测出来的rgb
    depth_final = torch.sum(weights * z_vals, -2) # [batch_size, num_rays, 1] 最终预测出来的深度

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    logging.info(f"fancy_integration output rgb_final.shape: {rgb_final.shape}")

    # 最终预测出来的rgb 生成最终的图像
    return rgb_final, depth_final, weights


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""
    """ return: 返回相机空间中的采样点，z_vals(深度)和光线方向。
        ray_start: 近平面
        ray_end: 远平面
    """

    W, H = resolution # (img_size, img_size)
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),  # (W, H)
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten() # (H*W,) 铺平
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan((2 * math.pi * fov / 360)/2) # (H*W,) 透视投影

    # 射线方向
    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1)) # (H*W, 3)


    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1) # (H*W, num_steps, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals # (H*W, num_steps, 3)

    points = torch.stack(n*[points]) # (n, H*W, num_steps, 3) n --> batch_size // batch_split
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device) # (n, H*W, 3)

    logging.info(f"get_initial_rays_trig's points.shape: {points.shape}, rays_d_cam.shape: {rays_d_cam.shape}")

    return points, z_vals, rays_d_cam  #TODO: debug these dimensions 

# 加入了随机扰动 对采样点引入随机噪声，最终points可以左右移动0.5初始点距
def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def transform_sampled_points(points, z_vals, ray_directions, device, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, 
                             v_mean=math.pi * 0.5, mode='normal'):
    """Samples a camera position and maps points in camera space to world space."""
    """ 采样相机位置并将相机空间中的点映射到世界空间。 """
    # n --> batch_size, num_rays --> H*W(pixels), num_steps --> num_samples
    n, num_rays, num_steps, channels = points.shape # input points.shape: [batch_size, num_rays, num_steps, 3]

    # TODO: the points's dims
    points, z_vals = perturb_points(points, z_vals, ray_directions, device)

    # 获取相机原点，水平角和仰视角 camera_origin.shape: [batch_size, 3], pitch.shape: [batch_size, 1], yaw.shape: [batch_size, 1]
    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev, vertical_stddev=v_stddev, 
                                                        horizontal_mean=h_mean, vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones((points.shape[0], points.shape[1], points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points # (n, num_rays, num_steps, 4)

    # should be n x 4 x 4 , n x r^2 x num_steps x 4 (采样点)
    transformed_points = torch.bmm(cam2world_matrix, # (n, num_rays, num_steps, 4)
                                   points_homogeneous.reshape(n, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)

    # 没有使用齐次坐标(向量的平移不变性) 射线方向
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3],  # (n, num_rays, 3(x,y,z))
                                           ray_directions.reshape(n, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(n, num_rays, 3)

    # 点需要平移，先转换成齐次坐标再作c2m 原点
    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device) # (n, 4, num_rays)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3] # (n, num_rays, 3(x,y,z)))

    # 返回转换之后的采样点，深度，光线方向，相机原点，仰视角，水平角
    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

# 在球面上以给定采样方式采样n个点，返回世界坐标下的相机位置和球坐标下的θ和φ
def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi) 水平角
    Phi is pitch in radians (0, pi) 仰视角
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5) # phi.shape [batch_size, 1]

    #TODO: what is the output_points?
    output_points = torch.zeros((n, 3), device=device) # output_points.shape [batch_size, 3]
    # world coordinate is the same as camera
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta) # x-axis
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta) # z-axis
    output_points[:, 1:2] = r*torch.cos(phi)                    # y-axis

    return output_points, phi, theta # [batch_size, 3], [batch_size, 1], [batch_size, 1]

# 逐相机生成c2w矩阵
def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix # (batch_size, 4, 4)

    return cam2world


# w2c矩阵
def create_world2cam_matrix(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam

# 分层采样第二步，根据coarse sample得到的weight重新精细采样
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples
