import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import logging


class Sine(nn.Module):
    """Sine Activation Function."""
    """ SINE 正弦周期激活函数 """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1) # 输入特征数量
            # uniform_() 方法将张量中的每个元素初始化为从均匀分布中获取的值
            # -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30  He正太初始化
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m): # 第一层 FiLM SIREN sin(γx + β) 初始化
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1) # num_input == 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)

 
def kaiming_leaky_init(m):  # 激活函数使用 leaky_relu 初始化
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

# Mapping Network 输入噪声 z 输出频率 γ 和相位 β
# class CustomMappingNetwork(nn.Module):     
#     def __init__(self, z_dim, map_hidden_dim, map_output_dim):
#         super().__init__()
#         """
#         output_dim = (8层隐藏层 + 1层额外层) * 256个维度 * 2个参数(频率 γ 和相位 β) (4608)
#         """
#         # Mapping network [256, 256] 3层
#         logging.info(f"mapping network: z_dim.shape: {z_dim}, map_hidden_dim.shape: {map_hidden_dim}, map_output_dim.shape: {map_output_dim}")
#         self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
#                                      nn.LeakyReLU(0.2, inplace=True),

#                                     nn.Linear(map_hidden_dim, map_hidden_dim),
#                                     nn.LeakyReLU(0.2, inplace=True),

#                                     nn.Linear(map_hidden_dim, map_hidden_dim),
#                                     nn.LeakyReLU(0.2, inplace=True),

#                                     nn.Linear(map_hidden_dim, map_output_dim))

#         self.network.apply(kaiming_leaky_init)
#         with torch.no_grad():
#             self.network[-1].weight *= 0.25

# change
class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, n_blocks=3):
        super().__init__()
        self.network = [nn.Linear(z_dim, map_hidden_dim),
                        nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(n_blocks):
            self.network.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.network.append(nn.Linear(map_hidden_dim, map_output_dim))
        self.network = nn.Sequential(*self.network)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25
                
    # 一次性计算出所有层的频率 γ 和相位 β
    def forward(self, z):
        frequencies_offsets = self.network(z) # [batch_size, 4608] 4608 = (8+1)*256*2
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2] # [batch_size, 2304]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:] # [batch_size, 2304]

        return frequencies, phase_shifts

# 初始化线性层的权重
def frequency_init(freq):
    def init(m):
         with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init
 
class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 初始化线性层
        self.layer = nn.Linear(input_dim, hidden_dim)   # 初始化Linear层 Linear --> FiLMLayer
    # x 输入的位置坐标信息 position x
    def forward(self, x, freq, phase_shift):
        logging.info(f"FiLMLayer input_x.shape: {x.shape}")
        x = self.layer(x)       # 计算线性层  x.shape [batch_size, num_rays*num_step, hidden_dim]  layer --> Linear FiLMLayer input
        logging.info(f"FiLMLayer output_x.shape: {x.shape}")
        # 通过激活函数计算 FiLM SIREN sin(γx + β)
        freq = freq.unsqueeze(1).expand_as(x) #  freq.shape [batch_size, num_rays*num_step, hidden_dim]
        phase_shift = phase_shift.unsqueeze(1).expand_as(x) # phase_shift.shape [batch_size, num_rays*num_step, hidden_dim]

        return torch.sin(freq * x + phase_shift)  # [batch_size, num_rays*num_step, hidden_dim] FiLM SIREN sin(γ(wx+b) + β)

# change start
class ReLULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer(x)
        return F.relu(x)

# 位置编码
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    # 创建编码函数列表和计算编码维度
    def create_embedding_fn(self): 
        embed_fns = []                  # 存储编码函数的列表
        d = self.kwargs['input_dims']   # 输入数据的维度
        out_dim = 0                     # 编码后的总维度
        if self.kwargs['include_input']:    # 为真，最终的编码结果包含原始坐标
            embed_fns.append(lambda x : x)  # 把一个不对数据做出改变的匿名函数添加到列表中
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2'] # 位置编码的最大频率 L-1 （0-9）
        N_freqs = self.kwargs['num_freqs']      # 位置编码的频率数量，论文中编码公式中的L
        
        if self.kwargs['log_sampling']:       # 正弦和余弦函数的频率值 A 
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)    # 指数增长
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)    # 线性增长
            
        '''
        位置编码公式 [sin, cos]  sin(2^0 * Πx), cos(2^0 * Πx).
        sin(x * freq), cos(x * freq)
        x: 坐标(x, y, z), freq: 2^L (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) L -> (0-9)
        '''
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        # 对输入数据进行位置编码，返回编码结果
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# 位置编码参数设置
def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
                'include_input' : False,         # 为真，最终的编码结果包含原始坐标
                'input_dims' : 3,               # 输入数据的维度 
                'max_freq_log2' : multires-1,   
                'num_freqs' : multires,         # 位置编码的频率数量，论文中编码公式中的L
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x) # 位置编码函数
    print(embed, embedder_obj.out_dim)
    return embed, embedder_obj.out_dim

# change end

# class TALLSIREN(nn.Module):
#     """Primary SIREN  architecture used in pi-GAN generators."""
#     """ 用于pi-GAN 生成器的主要SIREN架构。"""
#     def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
#         super().__init__()
#         self.device = device
#         self.input_dim = input_dim # 3 (x, y, z)张图片初始化
#         self.z_dim = z_dim # 256
#         self.hidden_dim = hidden_dim # 256
#         self.output_dim = output_dim # 4   
#         # TALLSIREN: input_dim.shape: 3, output_dim.shape: 4
#         logging.info(f"TALLSIREN: input_dim.shape: {input_dim}, output_dim.shape: {output_dim}")
#         self.network = nn.ModuleList([  # 8个FiLM SIREN 层 [3, 256], [256, 256] ... [256, 256]
#             FiLMLayer(input_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#         ])
#         self.final_layer = nn.Linear(hidden_dim, 1) # [256, 1] alpha输出层

#         self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)   # 加 ray direction d [256+3, 256] 
#         self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid()) # c(x, d) [256, 3] rgb输出层 普通线性层

#         self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) # mapping network output_dim = (8+1)*256*2

#         # 一次 25 张图片初始化
#         self.network.apply(frequency_init(25)) # 8 层 FiLMLayer 进行初始化
#         self.final_layer.apply(frequency_init(25))  # alpha 输出层初始化
#         self.color_layer_sine.apply(frequency_init(25)) # rgb 额外层初始化
#         self.color_layer_linear.apply(frequency_init(25))   # rgb 输出层初始化
#         self.network[0].apply(first_layer_film_sine_init) # 第一层 FiLMLayer 进行初始化

#     # input -> transformed_points (generator.py -> coarse_output) 
#     def forward(self, input, z, ray_directions, **kwargs):
#         frequencies, phase_shifts = self.mapping_network(z) # 从mapping network中获取频率 γ 和相位 β
#         return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

#     # SIREN MLP 网络 计算 SIREN sin(γx + β) 输出RGB 和 alpha
#     def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs): # MLP 计算
      
#         logging.info(f"forward_with_frequencies_phase_shifts input.shape: {input.shape}")
#         frequencies = frequencies*15 + 30
#         # x.shape [batch_size, num_rays*num_steps, 3] input -> points
#         x = input

#         for index, layer in enumerate(self.network): # 8层隐藏层的计算
#             # layer == FiLMLayer(i)   FiLM SIREN sin(γx + β)
#             # 每次取一层的 γ 和 β (end - start) == 256
#             start = index * self.hidden_dim
#             end = (index+1) * self.hidden_dim
#             # x.shape [batch_size, num_rays*num_steps, hidden_dim] 
#             x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end]) # -> FiLMLayer forward
#         # x通过8层 MLP计算后 的最终输出维度为 [batch_size, num_rays*num_steps, hidden_dim]
#         logging.info(f"forward_with_frequencies_phase_shifts after x.shape: {x.shape}")
#         sigma = self.final_layer(x) # sigma [batch_size, num_rays*num_steps, 1] alpha 输出层
#         rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1),  # ray_directions d [batch_size, num_rays*num_steps, 3]
#                                     frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:]) # 最后一层的 γ 和 β 259 -> 256
#         rbg = self.color_layer_linear(rbg) # rgb [batch_size, num_rays*num_steps] 输出层 256 -> 3

#         # 生成图片的时候需要将 alpha 和 rgb 拼接在一起 然后输入到volume rendering中渲染
#         return torch.cat([rbg, sigma], dim=-1) # return [batch_size, num_rays*num_steps, 4] rgb + alpha
    
# CHNAEG start


class UniformBoxWarp(nn.Module):    # 用于将输入的坐标映射到 [-1, 1]
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor

class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""
    """ 用于pi-GAN 生成器的主要SIREN架构。"""
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim # 3 (x, y, z)
        self.z_dim = z_dim # 256
        self.hidden_dim = hidden_dim # 256
        self.output_dim = output_dim # 4   


        # TALLSIREN: input_dim.shape: 3, output_dim.shape: 4
        # logging.info(f"TALLSIREN: input_dim.shape: {input_dim}, output_dim.shape: {output_dim}")
        self.network = nn.ModuleList([  # 8个FiLM SIREN 层 [3, 256], [256, 256] ... [256, 256]
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, hidden_dim),
        ])

        self.network_mlp = nn.ModuleList([
            ReLULayer(hidden_dim + 60, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
            ReLULayer(hidden_dim, hidden_dim),
        ]

        )
        self.final_layer = nn.Linear(hidden_dim, 1) # [256, 1] alpha输出层

        # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)   # 加 ray direction d [256+3, 256] 
        self.color_layer_sine = nn.ModuleList([     # 256+32+3 --> 256
            FiLMLayer(hidden_dim + 32 + 3, hidden_dim),
            ReLULayer(hidden_dim+32+24, hidden_dim),  # 拼接相机direction*3、position encoding*32
            # FiLMLayer(hidden_dim, hidden_dim),
            # FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid()) # c(x, d) [256, 3] rgb输出层 普通线性层

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2) # mapping network output_dim = (8+1)*256*2

        # 一次 25 张图片初始化
        self.network.apply(frequency_init(25)) # 8 层 FiLMLayer 进行初始化
        self.final_layer.apply(frequency_init(25))  # alpha 输出层初始化
        self.color_layer_sine.apply(frequency_init(25)) # rgb 额外层初始化
        self.color_layer_linear.apply(frequency_init(25))   # rgb 输出层初始化
        self.network[0].apply(first_layer_film_sine_init) # 第一层 FiLMLayer 进行初始化

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)

        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        # self.gridwarper = UniformBoxWarp(0.24)

    # input -> transformed_points (generator.py -> coarse_output) 
    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z) # 从mapping network中获取频率 γ 和相位 β
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    # SIREN MLP 网络 计算 SIREN sin(γx + β) 输出RGB 和 alpha
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs): # MLP 计算
      
        embed_multires = 10
        embed_multires_view = 4
        # logging.info(f"forward_with_frequencies_phase_shifts input.shape: {input.shape}")
        frequencies = frequencies*15 + 30
        # input = self.gridwarper(input)

        # 位置编码
        embed_fn, input_ch = get_embedder(embed_multires)
        embeddirs_fn, input_ch_views = get_embedder(embed_multires_view)
        # 双线性插值采样
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        # x.shape [batch_size, num_rays*num_steps, 3] input -> points
        x = input

        # 计算FiLM-MLP的颜色和密度
        for index, layer in enumerate(self.network): # 8层隐藏层的计算
            # layer == FiLMLayer(i)   FiLM SIREN sin(γx + β)
            # 每次取一层的 γ 和 β (end - start) == 256
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            # x.shape [batch_size, num_rays*num_steps, hidden_dim] 
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
            
        # rbg_film = torch.cat([x, shared_features], dim=-1)
        sigma_film = self.final_layer(x)
        rbg_film = self.color_layer_sine[0](torch.cat([x, shared_features, ray_directions], dim=-1))
        rbg_film = self.color_layer_linear(rbg_film)

        # for index, layer in enumerate(self.network): # 8层隐藏层的计算
        #     # layer == FiLMLayer(i)   FiLM SIREN sin(γx + β)
        #     # 每次取一层的 γ 和 β (end - start) == 256
        #     if index < 4:
        #         start = index * self.hidden_dim
        #         end = (index+1) * self.hidden_dim
        #         # x.shape [batch_size, num_rays*num_steps, hidden_dim] 
        #         x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end]) # -> FiLMLayer forward
        #     # 后四层为 ReLU MLP
        #     elif index == 4:
        #         embed_fn, input_ch = get_embedder(embed_multires)
        #         embeddirs_fn, input_ch_views = get_embedder(embed_multires_view)
        #         x = torch.cat([embed_fn(input), x], dim=-1)
        #         x = layer(x)
        #     else:
        #         start = index * self.hidden_dim
        #         end = (index+1) * self.hidden_dim
        #         x = layer(x)
        # sigma torch.Size([12, 24576, 1]) [batch_size, img_size*img_size*nums_stpes, 1]
        # for index, layer in enumerate(self.color_layer_sine):
        #     start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
        #     rbg = layer(rbg)
        # rbg = self.color_layer_linear(rbg) # rgb [batch_size, num_rays*num_steps] 输出层 256 -> 3

        # 计算ReLU-MLP的颜色和密度
        for index, layer in enumerate(self.network_mlp):
            if index == 0:
                x = torch.cat([embed_fn(input), x], dim=-1)
                x = layer(x)
            else:
                x = layer(x)
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine[1](torch.cat([x, shared_features, embeddirs_fn(ray_directions)], dim=-1))
        rbg = self.color_layer_linear(rbg)

        # 生成图片的时候需要将 alpha 和 rgb 拼接在一起 然后输入到volume rendering中渲染
        return torch.cat([rbg_film, sigma_film]), torch.cat([rbg, sigma], dim=-1) # return [batch_size, num_rays*num_steps, 4] rgb + alpha
# CHNAEG end


class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""
    """ 增加一个缩放因子 将输入的坐标映射到 [-1, 1]"""
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        print("SPATIALSIRENBASELINE:" )
        print("FiLMLayer_input_dim: ", input_dim)
        print("FiLMLayer_hidden_dim: ", hidden_dim)
        print("FiLMLayer_output_dim: ", output_dim)

        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.
        # 多一个缩放因子 应该类似NDC
        self.gridwarper = UniformBoxWarp(0.24) 

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        # 对输入的数据先进行一个 [-1, 1] 的缩放
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), 
                                    frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        
        return torch.cat([rbg, sigma], dim=-1)
    
    
    
class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor

# 从三维网格中采样特征
def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)

    """
    coordinates = coordinates.float() # (batch_size, num_points_per_batch, 3)
    grid = grid.float()
    
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1), # (batch_size, channels, H, W, D)
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=True) # 双线性插值
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""
    """ 更小的架构，具有额外的编码3D。通常可以提供更好的细节。 """
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        print("EmbeddingPiGAN128: ")
        print("FiLMLayer_input_dim: ", input_dim)
        print("FiLMLayer_hidden_dim: ", hidden_dim) 
        print("FiLMLayer_output_dim: ", output_dim)

        self.network = nn.ModuleList([
            FiLMLayer(32 + 3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        print(self.network)

        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. 
        # For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        logging.info(f"forward_with_frequencies_phase_shifts input.shape: {input.shape}")
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)  # 多一个细节特征输入 双线性插值
        x = torch.cat([shared_features, input], -1)     # x == shared_features + input  上面的为 x == input
        logging.info(f"forward_with_frequencies_phase_shifts x.shape: {x.shape}")
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        logging.info(f"forward_with_frequencies_phase_shifts after x.shape: {x.shape}")
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), 
                                    frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)

# Change of EmbeddingPiGAN128
"""
修改 color_layer_sine 层 添加额外的 postion encoding * 32
修改 forward_with_frequencies_phase_shifts 先对输入进行 SIERN-based层 计算 
然后将输出结果与shared_features进行拼接 再传入 color_layer_sine 层
"""  

# class EmbeddingPiGAN128(nn.Module):
#     """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""
#     """ 更小的架构，具有额外的编码3D。通常可以提供更好的细节。 """
#     def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
#         super().__init__()
#         self.device = device
#         self.input_dim = input_dim
#         self.z_dim = z_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
        
#         print("EmbeddingPiGAN128: ")
#         print("FiLMLayer_input_dim: ", input_dim)
#         print("FiLMLayer_hidden_dim: ", hidden_dim) 
#         print("FiLMLayer_output_dim: ", output_dim)

#         self.network = nn.ModuleList([
#             FiLMLayer(32 + 3, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#         ])
#         print(self.network)

#         self.final_layer = nn.Linear(hidden_dim, 1)

#         # self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
#         self.color_layer_sine = nn.ModuleList([
#             FiLMLayer(hidden_dim+32, hidden_dim),  # 拼接相机direction*3、position encoding*32
#             FiLMLayer(hidden_dim, hidden_dim),
#             FiLMLayer(hidden_dim, hidden_dim),
#         ])

#         self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

#         self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

#         self.network.apply(frequency_init(25))
#         self.final_layer.apply(frequency_init(25))
#         self.color_layer_sine.apply(frequency_init(25))
#         self.color_layer_linear.apply(frequency_init(25))
#         self.network[0].apply(modified_first_sine_init)

#         self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
#         # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
#         # a box of side-length 0.24, since the camera has such a narrow FOV. 
#         # For other scenes, with higher FOV, probably needs to be bigger.
#         self.gridwarper = UniformBoxWarp(0.24)

#     def forward(self, input, z, ray_directions, **kwargs):
#         frequencies, phase_shifts = self.mapping_network(z)
#         return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

#     def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
#         frequencies = frequencies*15 + 30

#         input = self.gridwarper(input)
#         shared_features = sample_from_3dgrid(input, self.spatial_embeddings)  # 多一个细节特征输入 双线性插值
#         logging.info(f"forward_with_frequencies_phase_shifts input.shape: {input.shape}")
#         x = input
#         logging.info(f"forward_with_frequencies_phase_shifts input.shape: {input.shape}")
#         for index, layer in enumerate(self.network):
#             start = index * self.hidden_dim
#             end = (index+1) * self.hidden_dim
#             x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

#         rbg = torch.cat([ray_directions, shared_features, x], -1)    
#         sigma = self.final_layer(x)
#         for index, layer in enumerate(self.color_layer_sine):
#             start, end = index * self.hidden_dim, (index+1) * self.hidden_dim
#             rbg = layer(rbg, frequencies[..., start:end], phase_shifts[..., start: end])
        
#         rbg = torch.sigmoid(self.color_layer_linear(rbg))

#         return torch.cat([rbg, sigma], dim=-1)


class EmbeddingPiGAN256(EmbeddingPiGAN128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64)*0.1)