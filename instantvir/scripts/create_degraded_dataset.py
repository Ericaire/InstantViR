import argparse
import os
import sys
import torch
import lmdb
from tqdm import tqdm
import numpy as np

from omegaconf import OmegaConf

# Assuming the project structure allows this import path
# Prefer lightweight VAE load; fall back to full pipeline only if available
try:
    from instantvir.models.wan.causal_inference import InferencePipeline  # heavy, may require flash-attn
    _HAS_PIPELINE = True
except Exception:
    InferencePipeline = None
    _HAS_PIPELINE = False
from instantvir.ode_data.create_lmdb_iterative import store_arrays_to_lmdb, get_array_shape_from_lmdb, retrieve_row_from_lmdb
import torch.nn.functional as F

# ---------------- Debug helpers ----------------
def _gpu_mem(prefix: str) -> None:
    try:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            alloc = torch.cuda.memory_allocated(dev) / (1024**2)
            rsv = torch.cuda.memory_reserved(dev) / (1024**2)
            print(f"[DEBUG][{prefix}] cuda allocated={alloc:.1f} MiB reserved={rsv:.1f} MiB")
    except Exception as e:
        print(f"[DEBUG] _gpu_mem error: {e}")

# ---------------- Frames source helpers ----------------
def _list_frames(frames_dir: str):
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png") and not f.startswith("._")])
    return [os.path.join(frames_dir, f) for f in files]

def _load_frames_btchw(frames_dir: str) -> torch.Tensor:
    from PIL import Image
    paths = _list_frames(frames_dir)
    if not paths:
        raise RuntimeError(f"No png frames found in {frames_dir}")
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            imgs.append(np.array(im))
    vid = np.stack(imgs, axis=0).transpose(0,3,1,2).astype(np.float32) / 255.0  # (T,C,H,W) in [0,1]
    vid = vid * 2.0 - 1.0  # [-1,1]
    return torch.from_numpy(vid).unsqueeze(0)  # (1,T,C,H,W)

# Reuse DAVI/DPS Resizer for faithful SR downsampling
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DAVI_UTIL_PATH = os.path.join(REPO_ROOT, "baseline", "diffusion-posterior-sampling")
if DAVI_UTIL_PATH not in sys.path:
    sys.path.append(DAVI_UTIL_PATH)
try:
    from util.resizer import Resizer  # type: ignore
except Exception as e:
    Resizer = None  # Fallback handled at callsite


def add_gaussian_noise_video(video: torch.Tensor, sigma: float = 0.2) -> torch.Tensor:
    """
    Adds Gaussian noise to a video tensor (frame by frame).
    
    Args:
        video (torch.Tensor): A video tensor of shape [B, T, C, H, W] in range [-1, 1].
        sigma (float): The standard deviation of the Gaussian noise.
    
    Returns:
        torch.Tensor: The noisy video tensor, clamped to [-1, 1].
    """
    noise = torch.randn_like(video) * sigma
    noisy_video = video + noise
    return torch.clamp(noisy_video, -1, 1)


def apply_gaussian_blur_video(video: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply spatial Gaussian blur in pixel space, per-frame, per-channel.
    video: [B, T, C, H, W] in [-1,1] or [0,1]
    Returns same shape, same range, with replicate-padding.
    """
    B, T, C, H, W = video.shape
    device = video.device
    dtype = video.dtype
    x = video.view(B * T, C, H, W)
    # local 2D Gaussian kernel to avoid importing heavy wan modules
    ax = torch.arange(kernel_size, dtype=dtype, device=device) - (kernel_size - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum().clamp(min=1e-8)
    kernel2d = (g.view(-1, 1) @ g.view(1, -1))
    kernel2d = (kernel2d / kernel2d.sum().clamp(min=1e-8)).view(1, 1, kernel_size, kernel_size)
    kernel = kernel2d
    kernel_c = kernel.repeat(C, 1, 1, 1)
    pad = kernel_size // 2
    x_blur = F.conv2d(x, kernel_c, padding=pad, groups=C)
    return x_blur.view(B, T, C, H, W)


def _center_crop_to_multiple(video: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Center-crop spatial dims (H,W) so both are divisible by factor.
    video: [B, T, C, H, W]
    """
    B, T, C, H, W = video.shape
    H_new = (H // factor) * factor
    W_new = (W // factor) * factor
    if H_new == H and W_new == W:
        return video
    h0 = (H - H_new) // 2
    w0 = (W - W_new) // 2
    return video[:, :, :, h0:h0 + H_new, w0:w0 + W_new]


def apply_sr_downsample_video(video: torch.Tensor, downscale_factor: int = 4) -> torch.Tensor:
    """
    Downsample video in pixel space by an integer factor using DAVI Resizer (anti-aliased).
    Args:
        video: [B, T, C, H, W]
        downscale_factor: integer scale > 1
    Returns:
        [B, T, C, H/f, W/f]
    """
    assert downscale_factor >= 2, "downscale_factor should be >= 2"
    video = _center_crop_to_multiple(video, downscale_factor)
    if Resizer is None:
        # Conservative fallback: bilinear with antialias
        B, T, C, H, W = video.shape
        x = video.view(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=1.0 / downscale_factor, mode="bilinear", align_corners=False, antialias=True)
        _, _, Hn, Wn = x.shape
        return x.view(B, T, C, Hn, Wn)

    # Use Resizer across H and W dims (5D aware)
    scale = 1.0 / float(downscale_factor)
    in_shape = list(video.shape)
    resizer = Resizer(in_shape=in_shape, scale_factor=scale)
    return resizer(video)


def create_random_mask(h: int, w: int, ratio: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a binary mask of shape [H, W] with given keep ratio (1=keep, 0=mask-out).
    ratio is the masked ratio if we interpret 0s as removed pixels? Here we define ratio as masked portion.
    For inpainting we want mask_ratio=0.5 (mask out 50%). We'll return mask_keep = 1 - mask (keep 50%).
    """
    num = h * w
    k = int((1.0 - ratio) * num)  # keep = (1 - mask_ratio)
    # Create 1s and 0s then shuffle
    mask_flat = torch.zeros(num, device=device, dtype=dtype)
    if k > 0:
        mask_flat[:k] = 1
    perm = torch.randperm(num, device=device)
    mask_flat = mask_flat[perm]
    return mask_flat.view(h, w)

def _parse_roi_rects(roi_rects_str: str) -> list[tuple[int, int, int, int]]:
    """
    Parse ROI rectangles from a semicolon-separated string: "x,y,w,h; x,y,w,h"
    Returns a list of tuples (x, y, w, h)
    """
    rects: list[tuple[int, int, int, int]] = []
    if not roi_rects_str:
        return rects
    parts = [p.strip() for p in roi_rects_str.split(';') if p.strip()]
    for p in parts:
        xywh = [s.strip() for s in p.split(',')]
        if len(xywh) != 4:
            continue
        try:
            x, y, w, h = map(int, xywh)
            rects.append((x, y, w, h))
        except Exception:
            continue
    return rects

def create_roi_random_keep_mask(h: int, w: int, roi_rects: list[tuple[int, int, int, int]],
                                roi_mask_ratio: float, bg_mask_ratio: float,
                                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a binary keep-mask [H, W] with different random mask ratios inside ROI vs background.
    - roi_mask_ratio: masked ratio inside ROI (e.g., 0.75 -> keep 0.25)
    - bg_mask_ratio: masked ratio outside ROI
    """
    total_pixels = h * w
    # boolean ROI union mask
    roi_bool = torch.zeros((h, w), dtype=torch.bool, device=device)
    for x, y, ww, hh in roi_rects:
        x0 = max(0, min(w, x))
        y0 = max(0, min(h, y))
        x1 = max(0, min(w, x + ww))
        y1 = max(0, min(h, y + hh))
        if x1 > x0 and y1 > y0:
            roi_bool[y0:y1, x0:x1] = True
    # indices
    roi_inds = roi_bool.view(-1).nonzero(as_tuple=False).flatten()
    bg_inds = (~roi_bool).view(-1).nonzero(as_tuple=False).flatten()
    # compute keep counts
    k_roi = int((1.0 - float(roi_mask_ratio)) * roi_inds.numel()) if roi_inds.numel() > 0 else 0
    k_bg = int((1.0 - float(bg_mask_ratio)) * bg_inds.numel()) if bg_inds.numel() > 0 else 0
    # allocate flat keep-mask
    keep_flat = torch.zeros((total_pixels,), dtype=dtype, device=device)
    if roi_inds.numel() > 0 and k_roi > 0:
        perm = torch.randperm(roi_inds.numel(), device=device)
        pick = roi_inds[perm[:k_roi]]
        keep_flat[pick] = 1
    if bg_inds.numel() > 0 and k_bg > 0:
        perm = torch.randperm(bg_inds.numel(), device=device)
        pick = bg_inds[perm[:k_bg]]
        keep_flat[pick] = 1
    return keep_flat.view(h, w)


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel of shape [1, 1, K].
    """
    k = torch.arange(kernel_size, dtype=dtype, device=device)
    k = k - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (k / float(sigma)) ** 2)
    kernel = kernel / kernel.sum().clamp(min=1e-8)
    return kernel.view(1, 1, kernel_size)


def apply_temporal_gaussian_blur_video(video: torch.Tensor, kernel_size_t: int, sigma_t: float) -> torch.Tensor:
    """
    Apply temporal Gaussian blur along the time dimension in pixel space.
    video: [B, T, C, H, W] in [-1,1] or [0,1]
    Returns same shape, with replicate-padding along time.
    """
    B, T, C, H, W = video.shape
    device = video.device
    dtype = video.dtype
    # reshape to [B*C*H*W, 1, T]
    x = video.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, 1, T)
    kernel = _get_gaussian_kernel1d(kernel_size_t, sigma_t, dtype=dtype, device=device)
    pad = kernel_size_t // 2
    x_padded = F.pad(x, (pad, pad), mode='replicate')
    x_blur = F.conv1d(x_padded, kernel, groups=1)
    x_blur = x_blur.view(B, C, H, W, T).permute(0, 4, 1, 2, 3).contiguous()
    return x_blur


def apply_temporal_uniform_blur_video(video: torch.Tensor, kernel_size_t: int) -> torch.Tensor:
    """
    Apply temporal uniform (box) blur along time dimension in pixel space.
    video: [B, T, C, H, W] in [-1,1] or [0,1]
    """
    B, T, C, H, W = video.shape
    device = video.device
    dtype = video.dtype
    x = video.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, 1, T)
    k = int(kernel_size_t)
    kernel = torch.ones(1, 1, k, dtype=dtype, device=device) / float(k)
    pad = k // 2
    x_padded = F.pad(x, (pad, pad), mode='replicate')
    x_blur = F.conv1d(x_padded, kernel, groups=1)
    x_blur = x_blur.view(B, C, H, W, T).permute(0, 4, 1, 2, 3).contiguous()
    return x_blur


def main():
    parser = argparse.ArgumentParser(description="Create a degraded dataset for inverse problems.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the model config file.")
    parser.add_argument("--original_lmdb_path", type=str, required=False, help="Path to the original clean latents LMDB dataset.")
    parser.add_argument("--original_frames_dir", type=str, required=False, help="Path to frames dir (alternative to original_lmdb_path).")
    parser.add_argument("--new_lmdb_path", type=str, required=True, help="Path to save the new degraded latents LMDB dataset.")
    # VAE options
    parser.add_argument("--source_vae_type", type=str, default="wan", choices=["wan", "leanvae"], help="VAE type used in source LMDB (for decoding)")
    parser.add_argument("--vae_type", type=str, default="wan", choices=["wan", "leanvae"], help="VAE type for output LMDB (for encoding)")
    parser.add_argument("--leanvae_ckpt_path", type=str, default="LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt", help="Path to LeanVAE checkpoint")
    # Degradation options
    parser.add_argument("--degradation_type", type=str, default="gaussian_noise", choices=["gaussian_noise", "gaussian_blur", "inpainting", "super_resolution", "temporal_gaussian", "temporal_uniform"], help="Type of pixel-space degradation to apply.")
    parser.add_argument("--noise_sigma", type=float, default=0.2, help="Std of Gaussian noise (for gaussian_noise)")
    parser.add_argument("--blur_kernel_size", type=int, default=61, help="Kernel size for Gaussian blur (for gaussian_blur)")
    parser.add_argument("--blur_sigma", type=float, default=3.0, help="Sigma for Gaussian blur (for gaussian_blur)")
    parser.add_argument("--temporal_kernel_size", type=int, default=7, help="Temporal kernel size for Gaussian PSF (for temporal_gaussian)")
    parser.add_argument("--temporal_sigma", type=float, default=1.0, help="Temporal sigma for Gaussian PSF (for temporal_gaussian)")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Mask ratio for inpainting (mask out this portion of pixels)")
    parser.add_argument("--roi_rects", type=str, default=None, help="Semicolon-separated ROI rectangles 'x,y,w,h;...'; if set, use ROI/background different mask ratios.")
    parser.add_argument("--roi_mask_ratio", type=float, default=None, help="Masked ratio inside ROI (e.g., 0.75). Defaults to mask_ratio if None.")
    parser.add_argument("--bg_mask_ratio", type=float, default=None, help="Masked ratio outside ROI. Defaults to mask_ratio if None.")
    parser.add_argument("--downscale_factor", type=int, default=4, help="Downscale factor for super_resolution (e.g., 4)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process for a quick test. Processes all if None.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index (inclusive) for sharding the dataset.")
    
    args = parser.parse_args()

    # Dump args for debugging
    print("===== [DEBUG] Args =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("========================")

    # --- 1. Setup Models and Data ---
    print("--- Setting up models and data loaders ---")
    config = OmegaConf.load(args.config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] device={device}")
    _gpu_mem("start")

    # Load source VAE (for decoding source LMDB)
    print(f"--- Loading source VAE (type: {args.source_vae_type}) ---")
    if args.source_vae_type == "leanvae":
        from instantvir.models.leanvae_wrapper import LeanVAEWrapper
        source_vae_wrapper = LeanVAEWrapper(ckpt_path=args.leanvae_ckpt_path)
        source_vae_wrapper.to(device).eval()
        source_vae_model = source_vae_wrapper.model
        source_encode_fn = lambda x: source_vae_wrapper.encode_native(x, use_amp=True)
        source_decode_fn = lambda x: source_vae_wrapper.decode_native(x, use_amp=True)
    else:  # WAN VAE
        # Load the WAN VAE directly (avoid InferencePipeline which may load wrong VAE based on config)
        # Minimal VAE loader without importing full WAN wrappers
        from instantvir.models.wan.wan_base.modules.vae import _video_vae
        # constants copied from WanVAEWrapper to avoid importing it
        mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ], dtype=torch.float32, device=device)
        std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ], dtype=torch.float32, device=device)
        inv_std = 1.0 / std
        source_vae_model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().to(device)
        scale = [mean, inv_std]
        
        # WAN VAE encode/decode functions
        source_encode_fn = lambda x: source_vae_model.encode(x, scale)
        source_decode_fn = lambda x: source_vae_model.decode(x, scale)

    # Load target VAE (for encoding to output LMDB)
    print(f"--- Loading target VAE (type: {args.vae_type}) ---")
    _gpu_mem("before_target_vae")
    if args.vae_type == "leanvae":
        from instantvir.models.leanvae_wrapper import LeanVAEWrapper
        target_vae_wrapper = LeanVAEWrapper(ckpt_path=args.leanvae_ckpt_path)
        target_vae_wrapper.to(device).eval()
        target_vae_model = target_vae_wrapper.model
        target_encode_fn = lambda x: target_vae_wrapper.encode_native(x, use_amp=True)
        target_decode_fn = lambda x: target_vae_wrapper.decode_native(x, use_amp=True)
    elif args.vae_type == "wan" and args.source_vae_type == "wan":
        # Reuse source VAE if both are WAN
        target_vae_model = source_vae_model
        target_encode_fn = source_encode_fn
        target_decode_fn = source_decode_fn
    else:  # WAN VAE but different from source
        if _HAS_PIPELINE:
            target_vae_model = source_vae_model
            target_encode_fn = source_encode_fn
            target_decode_fn = source_decode_fn
        else:
            from instantvir.models.wan.wan_base.modules.vae import _video_vae
            mean = torch.tensor([
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
            ], dtype=torch.float32, device=device)
            std = torch.tensor([
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
            ], dtype=torch.float32, device=device)
            inv_std = 1.0 / std
            target_vae_model = _video_vae(
                pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                z_dim=16,
            ).eval().to(device)
            scale_target = [mean, inv_std]
            target_encode_fn = lambda x: target_vae_model.encode(x, scale_target)
            target_decode_fn = lambda x: target_vae_model.decode(x, scale_target)

    _gpu_mem("after_target_vae")

    # Determine source mode (LMDB vs Frames)
    use_frames = bool(args.original_frames_dir) and not bool(args.original_lmdb_path)
    if use_frames:
        print(f"[DEBUG] Source mode: FRAMES -> {args.original_frames_dir}")
        frame_list = _list_frames(args.original_frames_dir)
        print(f"[DEBUG] Found {len(frame_list)} frames")
        total_len = 1
        orig_env = None
        orig_latents_shape = None
    else:
        print(f"[DEBUG] Source mode: LMDB -> {args.original_lmdb_path}")
        if not args.original_lmdb_path:
            raise ValueError("Either --original_lmdb_path or --original_frames_dir must be provided.")
        # Open original LMDB directly to avoid heavy model imports in datasets
        orig_env = lmdb.open(args.original_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        orig_latents_shape = get_array_shape_from_lmdb(orig_env, 'latents')
        print(f"[DEBUG] latents header: {orig_latents_shape}")
        total_len = orig_latents_shape[0]
    
    # --- 2. Create New LMDB Environment ---
    print(f"--- Creating new LMDB environment at {args.new_lmdb_path} ---")
    os.makedirs(os.path.dirname(args.new_lmdb_path), exist_ok=True)
    map_size = 1024 * 1024 * 1024 * 1024 
    env = lmdb.open(args.new_lmdb_path, map_size=map_size)
    
    # --- 3. Process and Store Data ---
    print("--- Starting data processing loop ---")
    
    current_index = 0
    # total_len already determined above (frames:1 or from LMDB header)
    if not use_frames:
        total_len = orig_latents_shape[0]
    start = max(0, int(args.start_index))
    end = total_len if args.num_samples is None else min(total_len, start + int(args.num_samples))
    print(f"--- Processing range [{start}, {end}) out of {total_len} samples ---")
    with torch.no_grad():
        for i in tqdm(range(start, end)):
            print(f"[DEBUG] sample_idx={i}")
            _gpu_mem(f"loop_start_{i}")
            if use_frames:
                prompt = f"FRAMES_{os.path.basename(args.original_frames_dir.rstrip('/'))}"
                clean_video = _load_frames_btchw(args.original_frames_dir).to(device)  # [1,T,C,H,W]
                print(f"[DEBUG] clean_video {tuple(clean_video.shape)} {clean_video.dtype} range=[{clean_video.min():.3f},{clean_video.max():.3f}]")
            else:
                # Retrieve clean latent and prompt from original LMDB
                clean_latent_np = retrieve_row_from_lmdb(orig_env, 'latents', np.float16, i, shape=orig_latents_shape[1:])
                # If 5D [S,T,C,H,W], take last step; if 1D flattened, reshape to [T,C,H,W]
                if clean_latent_np.ndim == 5:
                    clean_latent_np = clean_latent_np[-1]
                elif clean_latent_np.ndim == 1:
                    # assume WAN default latent size 21x16x60x104
                    if clean_latent_np.size == 21*16*60*104:
                        clean_latent_np = clean_latent_np.reshape(21,16,60,104)
                    else:
                        raise ValueError(f"Unexpected flattened latent size: {clean_latent_np.size}")
                prompt = retrieve_row_from_lmdb(orig_env, 'prompts', str, i)
                clean_latent = torch.tensor(clean_latent_np, dtype=torch.float32, device=device).unsqueeze(0) # [1,T,C,H,W]

            # STEP 1: Decode latent to pixel space
            if not use_frames:
                print("[DEBUG] decode latent -> pixel ...")
                if args.source_vae_type == "leanvae":
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        clean_video = source_decode_fn(clean_latent)  # [1,T,C,H,W]
                else:
                    clean_latent_for_decode = clean_latent.permute(0, 2, 1, 3, 4)
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        clean_video_permuted = source_decode_fn(clean_latent_for_decode)  # [1,C,T,H,W]
                    clean_video = clean_video_permuted.permute(0, 2, 1, 3, 4)  # [1,T,C,H,W]
                print(f"[DEBUG] clean_video {tuple(clean_video.shape)} {clean_video.dtype}")
            else:
                print("[DEBUG] frames-mode: skip source decode")
            _gpu_mem(f"after_decode_{i}")

            # STEP 2: Re-encode clean video if target VAE is different from source
            if args.source_vae_type != args.vae_type:
                # Re-encode clean video to target VAE's latent space
                print(f"[DEBUG] re-encode clean to target VAE ({args.vae_type}) ...")
                if args.vae_type == "leanvae":
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        clean_latent = target_encode_fn(clean_video.permute(0, 2, 1, 3, 4))  # [1,T,16,Hl,Wl]
                else:
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        clean_latent_permuted = target_encode_fn(clean_video.permute(0, 2, 1, 3, 4))  # [1,16,T,Hl,Wl]
                    clean_latent = clean_latent_permuted.permute(0, 2, 1, 3, 4)  # [1,T,16,Hl,Wl]
                print(f"[DEBUG] clean_latent (target) {tuple(clean_latent.shape)} {clean_latent.dtype}")
            elif use_frames:
                # Frames-mode and same VAE type for source/target:
                # we still need to obtain the clean_latent to store in LMDB.
                print(f"[DEBUG] frames-mode: encode clean to target VAE ({args.vae_type}) ...")
                if args.vae_type == "leanvae":
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        clean_latent = target_encode_fn(clean_video.permute(0, 2, 1, 3, 4))  # [1,T,16,Hl,Wl]
                else:
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        clean_latent_permuted = target_encode_fn(clean_video.permute(0, 2, 1, 3, 4))  # [1,16,T,Hl,Wl]
                    clean_latent = clean_latent_permuted.permute(0, 2, 1, 3, 4)  # [1,T,16,Hl,Wl]
                print(f"[DEBUG] clean_latent (frames-mode target) {tuple(clean_latent.shape)} {clean_latent.dtype}")

            # STEP 3: Apply pixel-space degradation
            print(f"[DEBUG] degradation type: {args.degradation_type}")
            if args.degradation_type == "gaussian_blur":
                degraded_video = apply_gaussian_blur_video(clean_video, args.blur_kernel_size, args.blur_sigma)
                store_mask = None
            elif args.degradation_type == "gaussian_noise":
                degraded_video = add_gaussian_noise_video(clean_video, sigma=args.noise_sigma)
                store_mask = None
            elif args.degradation_type == "inpainting":
                B, T, C, H, W = clean_video.shape
                roi_rects = _parse_roi_rects(args.roi_rects) if args.roi_rects else []
                if roi_rects:
                    roi_ratio = float(args.roi_mask_ratio) if args.roi_mask_ratio is not None else float(args.mask_ratio)
                    bg_ratio = float(args.bg_mask_ratio) if args.bg_mask_ratio is not None else float(args.mask_ratio)
                    print(f"[DEBUG] inpainting with ROI: rects={roi_rects}, roi_mask_ratio={roi_ratio}, bg_mask_ratio={bg_ratio}")
                    mask_keep_hw = create_roi_random_keep_mask(
                        H, W, roi_rects, roi_ratio, bg_ratio,
                        device=clean_video.device, dtype=clean_video.dtype
                    )
                else:
                    mask_keep_hw = create_random_mask(H, W, ratio=args.mask_ratio, device=clean_video.device, dtype=clean_video.dtype) # [H,W] in {0,1}
                mask_keep_btchw = mask_keep_hw.view(1, 1, 1, H, W).expand(B, T, C, H, W)
                degraded_video = clean_video * mask_keep_btchw
                store_mask = mask_keep_hw.detach().cpu().numpy().astype(np.float16)  # store keep-mask
            elif args.degradation_type == "temporal_gaussian":
                degraded_video = apply_temporal_gaussian_blur_video(
                    clean_video, kernel_size_t=int(args.temporal_kernel_size), sigma_t=float(args.temporal_sigma)
                )
                store_mask = None
            elif args.degradation_type == "temporal_uniform":
                degraded_video = apply_temporal_uniform_blur_video(
                    clean_video, kernel_size_t=int(args.temporal_kernel_size)
                )
                store_mask = None
            else:  # super_resolution (downsample by factor, no mask)
                degraded_video = apply_sr_downsample_video(clean_video, downscale_factor=args.downscale_factor)
                store_mask = None
            degraded_video = degraded_video.clamp(-1, 1)
            print(f"[DEBUG] degraded_video {tuple(degraded_video.shape)} {degraded_video.dtype}")
            
            # STEP 4: Encode degraded video back to latent space
            # Handle different VAE types
            print("[DEBUG] encode degraded -> target latent ...")
            if args.vae_type == "leanvae":
                # LeanVAE: [1,T,C,H,W] pixel -> [1,C,T,H,W] -> encode -> [1,T,16,Hl,Wl] latent
                degraded_video_for_encode = degraded_video.permute(0, 2, 1, 3, 4)  # [1,C,T,H,W]
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    degraded_latent = target_encode_fn(degraded_video_for_encode)  # [1,T,16,Hl,Wl]
            else:
                # WAN VAE: [1,T,C,H,W] pixel -> [1,C,T,H,W] -> encode -> [1,16,T,Hl,Wl] -> [1,T,16,Hl,Wl] latent
                degraded_video_for_encode = degraded_video.permute(0, 2, 1, 3, 4)  # [1,C,T,H,W]
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                    degraded_latent_permuted = target_encode_fn(degraded_video_for_encode)  # [1,16,T,Hl,Wl]
                degraded_latent = degraded_latent_permuted.permute(0, 2, 1, 3, 4)  # [1,T,16,Hl,Wl]
            print(f"[DEBUG] degraded_latent {tuple(degraded_latent.shape)} {degraded_latent.dtype}")
            _gpu_mem(f"after_encode_{i}")
            
            # Prepare data for storage (float16 for space)
            clean_latent_np = clean_latent.squeeze(0).detach().cpu().numpy().astype(np.float16)
            degraded_latent_np = degraded_latent.squeeze(0).detach().cpu().numpy().astype(np.float16)
            
            data_to_store = {
                "clean_latent": np.array([clean_latent_np]),
                "degraded_latent": np.array([degraded_latent_np]),
                "prompts": np.array([prompt])
            }
            if store_mask is not None:
                data_to_store["inpainting_mask"] = np.array([store_mask])
            
            store_arrays_to_lmdb(env, data_to_store, start_index=current_index)
            current_index += 1
            print(f"[DEBUG] stored index={current_index-1}")
            _gpu_mem(f"loop_end_{i}")

    # --- 4. Finalize LMDB ---
    print("--- Finalizing LMDB with shape information ---")
    with env.begin(write=True) as txn:
       
        final_clean_shape = clean_latent_np.shape
        final_degraded_shape = degraded_latent_np.shape
        
        txn.put("clean_latent_shape".encode(), ",".join(map(str, (current_index,) + final_clean_shape)).encode())
        txn.put("degraded_latent_shape".encode(), ",".join(map(str, (current_index,) + final_degraded_shape)).encode())
        txn.put("prompts_shape".encode(), ",".join(map(str, (current_index, 1))).encode())
        if 'store_mask' in locals() and store_mask is not None:
            txn.put("inpainting_mask_shape".encode(), ",".join(map(str, (current_index,) + store_mask.shape)).encode())
            txn.put("mask_ratio".encode(), str(args.mask_ratio).encode())
        if args.degradation_type == "super_resolution":
            txn.put("sr_scale_factor".encode(), str(args.downscale_factor).encode())

    print(f"--- Successfully created degraded dataset with {current_index} samples. ---")

if __name__ == "__main__":
    main() 