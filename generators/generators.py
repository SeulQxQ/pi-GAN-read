"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
import time
import curriculums
from torch.cuda.amp import autocast

from .volumetric_rendering import *

import logging

# 传入 (SIREN, metadata['latent_dim'])
class ImplicitGenerator3d(nn.Module):
    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        self.z_dim = z_dim # 256 (latent_dim)
        self.siren = siren(output_dim=4, z_dim=self.z_dim, input_dim=3, device=None) # 初始化 SIREN -> siren.py
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        self.generate_avg_frequencies() # 求频率和相位的平均值 [1, 2304] 2304 = 256 * (8 + 1)
        

    def forward(self, z, img_size, fov, ray_start, ray_end, num_steps, 
                h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, 
                sample_dist=None, lock_view_dependence=False, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        从 噪声向量，渲染参数，相机分布 生成图像
        """
        # z.shape: (3, 256) [batch_size / batch_split, z_dim], num_rays = img_size * img_size
        batch_size = z.shape[0] 

        # Generate initial camera rays and sample points. 生成初始相机射线和采样点
        # 返回sample points, z_vals, ray directions batch_size, pixels, num_steps, 1
        with torch.no_grad():
            # 获取 采样点，z_vals(深度)和光线方向 position x
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size, num_steps, resolution=(img_size, img_size), device=self.device, 
                fov=fov, ray_start=ray_start, ray_end=ray_end) 
            # points_cam.shape: [batch_size, img_size*img_size, num_steps, 3(RGB)] z_vals.shape:  [batch_size, img_size*img_size, num_steps, 1] rays_d_cam.shape: [batch_size, img_size*img_size, num_steps, 3(xyz)]
            logging.info(f"generators forward points_cam.shape: {points_cam.shape}, rays_d_cam.shape: {rays_d_cam.shape}")

            # transform_sampled_points 对相机位置进行采样，并将相机空间坐标映射到世界空间坐标
            # 采样点，z_vals(深度)，光线方向，光线原点，俯仰角，偏航角 转换后的坐标
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
            transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, 
                                         h_mean=h_mean, v_mean=v_mean, device=self.device, mode=sample_dist) # 坐标系变换
            # trasformed_points.shape: [batch_size, num_rays, num_steps, 3(RGB)] z_vals.shape: [batch_size, num_rays, num_steps, 1] transformed_ray_directions.shape: [batch_size, num_rays, num_steps, 3(xyz)] 
            # transformed_ray_origins.shape: [batch_size, num_rays, num_steps, 3(xyz)] pitch.shape: [batch_size, num_rays, 1] yaw.shape: [batch_size, num_rays, 1]

            # 坐标系变换 从相机坐标系到世界坐标系
            # transformed_ray_directions_expanded 转换后的射线方向
            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2) # [batch_size, num_rays, num_steps, 1, 3]
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1) # [batch_size, num_rays, num_steps, num_steps, 3]
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3) # [batch_size, num_rays*num_steps, 3]

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

        # Model prediction on course points MLP 隐藏层计算 粗糙采样
        """
        输入：transformed_points [batch_size, num_rays*num_steps, 3(xyz)], z [batch_size, 256], ray_directions  [batch_size, num_rays*num_steps, 3(xyz)]
        输出：rgb aplha [batch_size, num_rays*num_steps, 4]
        """
        # input siren
        # coarse_output: [batch_size, num_rays*num_steps, 4] rgb + alpha
        coarse_output = self.siren(transformed_points, z, ray_directions=transformed_ray_directions_expanded) # -> siren.py TALLSIREN forward
        # Change start 
        coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4) # [batch_size, num_rays, num_steps, 4] 每条光线上的每个采样点

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample: # 半球采样
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3) # 每个光线上的每个采样点
                logging.info(f"generators forward transformed_points_reshape.shape: {transformed_points.shape}")
                # 从 fancy_integration 中获取 weights 权重 用来进行重要性采样（精细采样） 

                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], 
                                                  noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                #### Start new importance sampling 重要性采样
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach()

                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                # fine_points.shape [batch_size, num_rays, num_steps, 3]

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                    transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                # 精细网络 采样点
                

                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3) # [batch_size, num_rays*num_steps, 3]
                

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points 精细采样后的点在进行预测
            """
            输入：fine_points [batch_size, num_rays*num_steps, 3], z [batch_size, 246], ray_directions [batch_size, num_rays*num_steps, 3]
            输出：fine_output [batch_size, num_rays, nums_steps, 4](rgb aplha)
            """
            fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded)

            fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4) # [batch_size, num_rays, num_steps, 4]

            # Combine course and fine points 组合粗糙采样和精细采样
            # 最终输出： all_z_vals all_outputs

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2) # [batch_size, num_rays, num_steps*2, 4]

            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2) # [batch_size, num_rays, num_steps*2, 1]
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices) # [batch_size, num_rays, num_steps*2, 1]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4)) # [batch_size, num_rays, num_steps*2, 4]
        else:
            all_outputs = coarse_output

            all_z_vals_film = z_vals
            all_z_vals = z_vals


        # Create images with NeRF 
        # 使用 NeRF 创建图像
        # 输出：rgb [batch_size, num_rays, 3], depth [batch_size, num_rays, 1], weight [batch_size, num_rays, num_stpes*2 1]
        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, 
                                                   white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), 
                                                   clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])
        #  还原 pixels.shape: [batch_size, img_size, img_size, 3]
        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1 # 交换维度 [batch_size, 3, img_size, img_size]
        # pixels.shape: [batch_size, 3, img_size, img_size] pitch.shape: [batch_size, 1] yaw.shape: [batch_size, 1]
        return pixels, torch.cat([pitch, yaw], -1)


    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""
        """ 计算平均频率和相移 z_dim: 256"""
        z = torch.randn((10000, self.z_dim), device=self.siren.device) # 1w个随机噪声 都生成一个256维的随机向量 (10000, 256)
        
        # mapping_network 返回频率和相位 生成10000个随机 频率和相位 [10000, 2304]
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z) # siren.py -> CustomMappingNetwork forward
        self.avg_frequencies = frequencies.mean(0, keepdim=True)    # 求 频率 平均值 [1, 2304]
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)  # 求 相位 平均值 [1, 2304]
        return self.avg_frequencies, self.avg_phase_shifts


    def staged_forward(self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, 
                       psi=1, lock_view_dependence=False, max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2, 
                       sample_dist=None, hierarchical_sample=False, **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0] # 25

        # ADD：计算频率和相位的平均值
        self.generate_avg_frequencies()

        with torch.no_grad():

            # 从mapping network中获取频率和相位
            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (raw_phase_shifts - self.avg_phase_shifts)


            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), 
                                                                   device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
                transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, 
                                         v_mean=v_mean, device=self.device, mode=sample_dist)

            logging.info(f"generators staged_forward transformed_points.shape: {transformed_points.shape}")

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output_film = coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    # 获取粗糙采样的输出 [rgb alpha]
                    coarse_output_film[b:b+1, head:tail], coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                        transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], 
                        ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            logging.info(f"generators staged_forward no reshape coarse_output.shape: {coarse_output.shape}")
            coarse_output_film = coarse_output_film.reshape(batch_size, img_size * img_size, num_steps, 4)
            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)

            # 半球采样
            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights_film = fancy_integration(coarse_output_film, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], 
                                    noise_std=kwargs['nerf_noise'])
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], 
                                                      noise_std=kwargs['nerf_noise'])

                    weights_film = weights_film.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5

                    #### Start new importance sampling 重要性采样
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps)
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    
                    fine_z_vals_film = sample_pdf(z_vals_mid, weights_film[:, 1:-1], num_steps, det=False).detach().to(self.device)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], num_steps, det=False).detach().to(self.device)
                    
                    fine_z_vals_film = fine_z_vals_film.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points_film = transformed_ray_origins.unsqueeze(2).contiguous() + \
                        transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals_film.expand(-1,-1,-1,3).contiguous()
                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                        transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous()
                    
                    logging.info(f"generators staged_forward hierarchical_sample fine_points.shape: {fine_points.shape}")

                    fine_points_film = fine_points_film.reshape(batch_size, img_size*img_size*num_steps, 3)
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                # Sequentially evaluate siren with max_batch_size to avoid OOM 
                # 批量预测
                    
                fine_output_film = torch.zeros((batch_size, fine_points_film.shape[1], 4), device=self.device)
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        # 获取精炼采样的输出 [rgb alpha]
                        fine_output_film[b:b+1, head:tail], fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], 
                            ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size
                
                fine_output_film = fine_output_film.reshape(batch_size, img_size * img_size, num_steps, 4)
                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                
                all_outputs_film = torch.cat([fine_output_film, coarse_output_film], dim = -2)
                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)

                all_z_vals_film = torch.cat([fine_z_vals_film, z_vals], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)

                _, indices_film = torch.sort(all_z_vals_film, dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)

                all_z_vals_film = torch.gather(all_z_vals_film, -2, indices_film)
                all_z_vals = torch.gather(all_z_vals, -2, indices)

                all_outputs_film = torch.gather(all_outputs_film, -2, indices_film.expand(-1, -1, -1, 4))
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs_film = coarse_output_film
                all_outputs = coarse_output
                all_z_vals = z_vals

            # 使用NERF的体渲染计算每个像素的 RGB depth weight
            pixels_film, depth_film, weights_film = fancy_integration(all_outputs_film, all_z_vals_film, device=self.device,
                                                    white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], 
                                                    last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None),
                                                    noise_std=kwargs['nerf_noise'])
            
            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, 
                                                    white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], 
                                                    last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None),
                                                    noise_std=kwargs['nerf_noise'])
            
            depth_map_film = depth_film.reshape(batch_size, img_size, img_size).contiguous().cpu()
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()

            pixels_film = pixels_film.reshape((batch_size, img_size, img_size, 3))
            pixels_film = pixels_film.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        logging.info(f"generators staged_forward pixels.shape: {pixels.shape}")
        # 返回像素值和深度图
        return pixels, depth_map

    # Used for rendering interpolations
    def staged_forward_with_frequencies(self, truncated_frequencies, truncated_phase_shifts, img_size, 
                                        fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean, 
                                        psi=0.7, lock_view_dependence=False, max_batch_size=50000, depth_map=False, 
                                        near_clip=0, far_clip=2, sample_dist=None, hierarchical_sample=False, **kwargs):
        
        # TODO: batch_size's dims
        batch_size = truncated_frequencies.shape[0]
        logging.info(f"generators staged_forward_with_frequencies batch_size.shape: {batch_size.shape}")

        with torch.no_grad():
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), 
                                                                   device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
            
            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
                transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, 
                                         device=self.device, mode=sample_dist)


            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE batch size
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            # 批量预测
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    _, coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                        transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], 
                        ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            # END BATCHED SAMPLE

            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], 
                                                      noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                          transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        _, fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(
                            fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], 
                            ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                # END BATCHED SAMPLE

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, 
                                                       white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], 
                                                       last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), 
                                                       noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()


            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map


    def forward_with_frequencies(self, frequencies, phase_shifts, img_size, fov, ray_start, ray_end, num_steps, 
                                 h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, **kwargs):
        
        batch_size = frequencies.shape[0]
        logging.info(f"generators forward_with_frequencies batch_size.shape: {batch_size.shape}")

        points_cam, z_vals, rays_d_cam = get_initial_rays_trig(batch_size, num_steps, resolution=(img_size, img_size), 
                                                               device=self.device, fov=fov, ray_start=ray_start, ray_end=ray_end) # batch_size, pixels, num_steps, 1
        
        transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw = \
            transform_sampled_points(points_cam, z_vals, rays_d_cam, h_stddev=h_stddev, v_stddev=v_stddev, h_mean=h_mean, v_mean=v_mean, 
                                     device=self.device, mode=sample_dist)


        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1
            
        _, coarse_output = self.siren.forward_with_frequencies_phase_shifts(
            transformed_points, frequencies, phase_shifts, 
            ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
        
        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], 
                                                  noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1. Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + \
                    transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling
                
                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            _, fine_output = self.siren.forward_with_frequencies_phase_shifts(
                fine_points, frequencies, phase_shifts, 
                ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), 
                                                   last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], 
                                                   noise_std=kwargs['nerf_noise'])

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)
