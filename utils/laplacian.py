import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

def laplacian_of_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    assert size % 2 == 1, "Kernel size must be odd"
    ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    norm = (xx**2 + yy**2) / (2 * sigma**2)
    LoG = -1 / (math.pi * sigma**4) * (1 - norm) * torch.exp(-norm)
    LoG -= LoG.mean()  # normalize to zero-mean
    return LoG

def argmax_LoG_sigma(input_tensor: torch.Tensor, size_list: list):
    responses = []
    sigma_list = []
    for i, size in tqdm(enumerate(size_list)):
        sigma = (size - 1.) / 6.0
        sigma_list.append(sigma)
        kernel = laplacian_of_gaussian_kernel(size, sigma).to(input_tensor.device)
        kernel = kernel.view(1, 1, size, size)
        response = F.conv2d(input_tensor, kernel, padding=size // 2, groups=input_tensor.shape[1])
        if i == 0:
            responses.append(torch.ones_like(response) * 1e-2)
        else:
            responses.append(response.abs()*(sigma**2) - 2e-3*sigma)
            
    stacked = torch.stack(responses, dim=0)  # [S, B, C, H, W]

    S = stacked.shape[0]
    if S == 1:
        argmin_scale = torch.zeros(stacked.shape[1:], dtype=torch.long, device=stacked.device)
    else:
        R = stacked  # [S, B, C, H, W]

        local_min = torch.zeros_like(R, dtype=torch.bool)

        local_min[1:-1] = (R[1:-1] <= R[:-2]) & (R[1:-1] <= R[2:])

        local_min[0] = (R[0] <= R[1])
        local_min[-1] = (R[-1] < R[-2])

        lm_float = local_min.float()
        csum = torch.cumsum(lm_float, dim=0)                  
        first_lm_mask = local_min & (csum == 1)               

        first_idx = torch.argmax(first_lm_mask.float(), dim=0)  # [B, C, H, W]

        has_local = local_min.any(dim=0)                        # [B, C, H, W] bool
        global_min_idx = torch.argmin(R, dim=0)                 # [B, C, H, W] long

        argmin_scale = torch.where(has_local, first_idx, global_min_idx)

    sigma_tensor = torch.tensor(sigma_list, device=argmin_scale.device, dtype=torch.float32)
    sigma_map = sigma_tensor[argmin_scale]

    return sigma_map

