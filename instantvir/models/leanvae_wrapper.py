"""
LeanVAE Wrapper for InstantViR.
Provides frame-by-frame encode/decode to maintain temporal dimension compatibility with WAN VAE.
"""
import os
import sys
import torch
import torch.nn as nn
from typing import Optional

# Add LeanVAE to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LEANVAE_PATH = os.path.join(REPO_ROOT, "LeanVAE-master")
if LEANVAE_PATH not in sys.path:
    sys.path.append(LEANVAE_PATH)

from LeanVAE import LeanVAE
from instantvir.models.model_interface import VAEInterface


class LeanVAEWrapper(VAEInterface):
    """
    Wrapper for LeanVAE that provides frame-by-frame encoding/decoding.
    
    Key differences from WAN VAE:
    - Normalization: LeanVAE uses [-0.5, 0.5] vs WAN's learned mean/std
    - Latent channels: 16 (same as WAN 16-channel)
    - Spatial compression: 8x (same as WAN)
    - Temporal compression: DISABLED (process frame-by-frame to keep T unchanged)
    
    Input/Output shapes:
    - encode: pixel [B,C,T,H,W] → latent [B,T,16,H/8,W/8]
    - decode: latent [B,T,16,H/8,W/8] → pixel [B,T,C,H,W]
    """
    
    def __init__(self, ckpt_path: str = "LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt"):
        super().__init__()
        
        # Load LeanVAE model
        self.model = LeanVAE.load_from_checkpoint(ckpt_path, device='cpu', strict=False)
        self.model.eval().requires_grad_(False)
        
        # LeanVAE normalization range (from README)
        # Input/output pixels are normalized to [-0.5, 0.5]
        # This is different from WAN VAE's learned mean/std
        self.pixel_min = -0.5
        self.pixel_max = 0.5
        self.pixel_range = 1.0  # max - min
        
        # For compatibility with WAN VAE interface (not actually used, but kept for consistency)
        # LeanVAE doesn't use learned mean/std in latent space
        self.mean = torch.zeros(16)
        self.std = torch.ones(16)
    
    def to(self, *args, **kwargs):
        """Override to() to ensure the wrapped model is also moved"""
        super().to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)
        return self
        
    def normalize_pixel(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        Normalize pixel values from [-1, 1] (standard range) to [-0.5, 0.5] (LeanVAE range).
        """
        # Input: [-1, 1], Output: [-0.5, 0.5]
        # Preserve dtype by using tensor scalar
        return pixel * torch.tensor(0.5, dtype=pixel.dtype, device=pixel.device)
    
    def denormalize_pixel(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        Denormalize pixel values from [-0.5, 0.5] (LeanVAE range) to [-1, 1] (standard range).
        """
        # Input: [-0.5, 0.5], Output: [-1, 1]
        # Preserve dtype by using tensor scalar
        return pixel * torch.tensor(2.0, dtype=pixel.dtype, device=pixel.device)
    
    def encode_native(self, video: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
        """
        Encode video using LeanVAE's native temporal compression.
        
        Args:
            video: [B, C, T, H, W] in range [-1, 1], where T must be 4n+1
            use_amp: Use automatic mixed precision for faster encoding
            
        Returns:
            latent: [B, T/4+1, 16, H/8, W/8]
        """
        B, C, T, H, W = video.shape
        device = video.device
        
        # Verify T is 4n+1
        if (T - 1) % 4 != 0:
            raise ValueError(f"LeanVAE requires T to be 4n+1, got T={T}")
        
        # Normalize to LeanVAE range [-0.5, 0.5]
        video_norm = self.normalize_pixel(video)
        
        # Use LeanVAE's native encode with temporal compression
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    latent = self.model.encode(video_norm)  # [B, d, T/4+1, H/8, W/8]
            else:
                latent = self.model.encode(video_norm)
        
        # Permute to [B, T/4+1, d, H/8, W/8] for consistency with WAN VAE format
        latent = latent.permute(0, 2, 1, 3, 4)
        
        return latent
    
    def decode_native(self, latent: torch.Tensor, use_amp: bool = True) -> torch.Tensor:
        """
        Decode latent using LeanVAE's native temporal decompression.
        
        Args:
            latent: [B, T/4+1, 16, H/8, W/8]
            use_amp: Use automatic mixed precision for faster decoding
            
        Returns:
            video: [B, T, C, H, W] in range [-1, 1], where T = (T_latent-1)*4+1
        """
        B, T_latent, d, Hl, Wl = latent.shape
        device = latent.device
        
        # Permute to [B, d, T_latent, Hl, Wl] for LeanVAE
        latent_perm = latent.permute(0, 2, 1, 3, 4)
        
        # Use LeanVAE's native decode with temporal decompression
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    video_norm = self.model.decode(latent_perm)  # [B, C, T, H, W]
            else:
                video_norm = self.model.decode(latent_perm)
        
        # Clamp to LeanVAE range and denormalize to [-1, 1]
        video_norm = video_norm.clamp(self.pixel_min, self.pixel_max)
        video = self.denormalize_pixel(video_norm)
        
        # Permute to [B, T, C, H, W] for consistency with WAN VAE
        video = video.permute(0, 2, 1, 3, 4)
        
        return video
    
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        VAEInterface required method: decode latent to pixel space.
        
        Args:
            latent: [B, T, C, H, W] where C=16 (latent channels)
            
        Returns:
            video: [B, T, C, H, W] where C=3 (RGB channels) in range [-1, 1]
        """
        # Check if AMP is enabled via environment variable (matching WAN VAE behavior)
        use_amp = os.environ.get('VAE_DECODE_AMP', '0') == '1'
        return self.decode_native(latent, use_amp=use_amp)
    
    def encode_to_latent(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel video to latent space (for preprocessing scripts).
        
        Args:
            video: [B, T, C, H, W] in range [-1, 1], where T must be 4n+1
            
        Returns:
            latent: [B, T/4+1, 16, H/8, W/8]
        """
        # Check if AMP is enabled via environment variable
        use_amp = os.environ.get('VAE_ENCODE_AMP', '0') == '1'
        
        # Permute to [B, C, T, H, W] for encode_native
        video_perm = video.permute(0, 2, 1, 3, 4)
        latent = self.encode_native(video_perm, use_amp=use_amp)
        
        return latent
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space (alias for encode_to_latent).
        
        Args:
            video: [B, T, C, H, W] in range [-1, 1]
            
        Returns:
            latent: [B, T/4+1, 16, H/8, W/8]
        """
        return self.encode_to_latent(video)
    
    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to video (alias for decode_to_pixel).
        
        Args:
            latent: [B, T, 16, H/8, W/8]
            
        Returns:
            video: [B, T, C, H, W] in range [-1, 1]
        """
        return self.decode_to_pixel(latent)
    
    def forward(self, x: torch.Tensor, mode: str = 'decode') -> torch.Tensor:
        """
        Forward pass for compatibility.
        
        Args:
            x: Input tensor
            mode: 'encode' or 'decode'
            
        Returns:
            Output tensor
        """
        if mode == 'encode':
            return self.encode_to_latent(x)
        elif mode == 'decode':
            return self.decode_to_pixel(x)
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'encode' or 'decode'.")

