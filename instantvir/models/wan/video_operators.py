import torch
import torch.nn.functional as F
from torchvision import transforms
import math
from einops import rearrange, repeat
import random


def _get_gaussian_kernel2d(kernel_size, sigma, dtype, device):
    """Create a 2D Gaussian kernel."""
    k = torch.arange(kernel_size, dtype=dtype, device=device)
    k -= kernel_size // 2
    x, y = torch.meshgrid(k, k, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

def spatial_blur_latent(x, kernel_size_s=7, sigma_s=1.5):
    """
    Apply spatial Gaussian blur in latent space, frame by frame, with full dtype control.
    x: [B, T, C, H, W] latent tensor
    kernel_size_s: size of spatial blur kernel
    sigma_s: standard deviation of spatial blur kernel
    """
    B, T, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # Reshape to treat each frame as an image in a batch
    x_reshaped = x.view(B * T, C, H, W)
    
    # Create Gaussian kernel with the correct dtype
    kernel = _get_gaussian_kernel2d(kernel_size_s, sigma_s, dtype=dtype, device=device)
    
    # Apply convolution channel-wise for each frame
    # To do this, we set groups=C in conv2d
    # The kernel needs to be [C, 1, H, W]
    kernel_c = kernel.repeat(C, 1, 1, 1)
    
    padding = kernel_size_s // 2
    x_blurred_reshaped = F.conv2d(x_reshaped, kernel_c, padding=padding, groups=C)
    
    # Reshape back to the original video format
    x_blurred = x_blurred_reshaped.view(B, T, C, H, W)
    
    return x_blurred


def temporal_blur_latent(x, kernel_size_t=7):
    """
    Apply temporal blur in latent space
    x: [B, T, C, H, W] latent tensor
    kernel_size_t: size of temporal blur kernel
    
    This implements a simple temporal averaging filter in latent space.
    Since we're in latent space, we can't directly use the pixel-space blur
    from SVI/measurements.py, but we can approximate temporal blur by
    averaging adjacent frames in the latent space.
    """
    B, T, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    
    # Create a uniform temporal kernel
    kernel = torch.ones(kernel_size_t, dtype=dtype, device=device)
    kernel = kernel / kernel.sum()
    
    # Reshape kernel for conv1d: [out_channels=1, in_channels=1, kernel_size]
    kernel = kernel.view(1, 1, -1)
    
    # Apply temporal blur channel-wise
    # First reshape x to [B*C*H*W, 1, T] for conv1d
    x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, C, H, W, T]
    x_reshaped = x_reshaped.view(B * C * H * W, 1, T)
    
    # Apply padding (replicate edge frames)
    padding = kernel_size_t // 2
    x_padded = F.pad(x_reshaped, (padding, padding), mode='replicate')
    
    # Apply convolution
    x_blurred = F.conv1d(x_padded, kernel, groups=1)
    
    # Reshape back to original format
    x_blurred = x_blurred.view(B, C, H, W, T)
    x_blurred = x_blurred.permute(0, 4, 1, 2, 3).contiguous()  # [B, T, C, H, W]
    
    return x_blurred


def temporal_gaussian_blur_latent(x: torch.Tensor, kernel_size_t: int = 7, sigma_t: float = 1.0) -> torch.Tensor:
    """
    Apply temporal Gaussian blur along time in latent space.
    x: [B, T, C, H, W]
    """
    B, T, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    # reshape to [B*C*H*W, 1, T]
    x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, 1, T)
    k = torch.arange(kernel_size_t, dtype=dtype, device=device) - (kernel_size_t - 1) / 2
    g = torch.exp(-0.5 * (k / float(sigma_t)) ** 2)
    g = g / g.sum().clamp(min=1e-8)
    kernel = g.view(1, 1, kernel_size_t)
    pad = kernel_size_t // 2
    x_padded = F.pad(x_reshaped, (pad, pad), mode='replicate')
    x_blurred = F.conv1d(x_padded, kernel, groups=1)
    x_blurred = x_blurred.view(B, C, H, W, T).permute(0, 4, 1, 2, 3).contiguous()
    return x_blurred


def temporal_uniform_blur_latent(x: torch.Tensor, kernel_size_t: int = 7) -> torch.Tensor:
    """
    Apply temporal uniform (box) blur along time in latent space.
    x: [B, T, C, H, W]
    """
    B, T, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    # reshape to [B*C*H*W, 1, T]
    x_reshaped = x.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, 1, T)
    # box kernel
    k = int(kernel_size_t)
    assert k > 0 and k % 1 == 0
    kernel = torch.ones(1, 1, k, dtype=dtype, device=device) / float(k)
    pad = k // 2
    x_padded = F.pad(x_reshaped, (pad, pad), mode='replicate')
    x_blurred = F.conv1d(x_padded, kernel, groups=1)
    x_blurred = x_blurred.view(B, C, H, W, T).permute(0, 4, 1, 2, 3).contiguous()
    return x_blurred


def generate_inpainting_mask(latents: torch.Tensor, mask_type: str, box_size: list[int]) -> torch.Tensor:
    """
    Generates a mask for inpainting. 1s for known areas, 0s for unknown (to be inpainted).

    Args:
        latents (torch.Tensor): The latent tensor of shape [B, T, C, H, W].
        mask_type (str): "center" or "random".
        box_size (list[int]): [box_h, box_w] for the mask.

    Returns:
        torch.Tensor: A mask of shape [B, T, 1, H, W].
    """
    B, T, C, H, W = latents.shape
    mask = torch.ones(B, T, 1, H, W, device=latents.device, dtype=latents.dtype)
    box_h, box_w = box_size

    if box_h >= H or box_w >= W:
        raise ValueError("Box size cannot be larger than the latent dimensions.")

    if mask_type == "center":
        h_start = (H - box_h) // 2
        w_start = (W - box_w) // 2
    elif mask_type == "random":
        h_start = random.randint(0, H - box_h)
        w_start = random.randint(0, W - box_w)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    mask[:, :, :, h_start : h_start + box_h, w_start : w_start + box_w] = 0
    return mask


def add_noise_latent(x: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    Adds Gaussian noise to a latent tensor.
    x: [B, T, C, H, W] latent tensor
    noise_level: standard deviation of noise
    """
    noise = torch.randn_like(x) * noise_level
    return x + noise


def random_mask_latent(x, mask_ratio=0.5):
    """
    Apply random masking to latent tensor
    x: [B, T, C, H, W] latent tensor
    mask_ratio: ratio of pixels to mask
    """
    B, T, C, H, W = x.shape
    # Create random mask
    mask = torch.rand(B, T, 1, H, W, device=x.device) > mask_ratio
    return x * mask


def downsample_latent(x, scale_factor=0.25):
    """
    Downsample latent tensor spatially
    x: [B, T, C, H, W] latent tensor
    scale_factor: downsampling factor (< 1)
    """
    # Flatten temporal dimension
    B, T, C, H, W = x.shape
    x_flat = x.view(B * T, C, H, W)
    
    # Downsample
    x_down = F.interpolate(x_flat, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    
    # Reshape back
    _, _, H_new, W_new = x_down.shape
    x_down = x_down.view(B, T, C, H_new, W_new)
    
    return x_down