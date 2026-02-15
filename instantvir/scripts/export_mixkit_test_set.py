import argparse
import os
from omegaconf import OmegaConf
import torch
import numpy as np
from tqdm import tqdm
from diffusers.utils import export_to_video
from torchvision.utils import save_image

from instantvir.data import InverseProblemLMDBDataset
from instantvir.models.wan.causal_inference import InferencePipeline
from torch.utils.data import random_split
import torch.nn.functional as F


def gaussian_blur_video(video_btchw: torch.Tensor, sigma: float = 1.5, kernel_size: int = 7) -> torch.Tensor:
    B, T, C, H, W = video_btchw.shape
    device = video_btchw.device
    dtype = video_btchw.dtype
    k = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    x, y = torch.meshgrid(k, k, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)
    video = video_btchw.view(B*T, C, H, W)
    pad = kernel_size // 2
    blurred = F.conv2d(video, kernel, padding=pad, groups=C)
    return blurred.view(B, T, C, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--kernel", type=int, default=7)
    parser.add_argument("--max_videos", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config_path)

    pipeline = InferencePipeline(config, device=device)
    # Ensure all modules (including VAE) are on the right device/dtype
    pipeline.to(device=device, dtype=torch.float32)

    full_dataset = InverseProblemLMDBDataset(
        data_path=args.data_path,
        inverse_problem_type="spatial_blur",
        degradation_params={}
    )
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    total = len(val_dataset) if args.max_videos < 0 else min(args.max_videos, len(val_dataset))

    for i in tqdm(range(total)):
        sample = val_dataset[i]
        clean_latent = sample["clean_latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
        prompt = sample["prompts"]

        with torch.no_grad():
            video = pipeline.vae.decode_to_pixel(clean_latent)
            video = (video * 0.5 + 0.5).clamp(0, 1)

        blurred = gaussian_blur_video(video, sigma=args.sigma, kernel_size=args.kernel)

        vid_np = video[0].permute(0, 2, 3, 1).cpu().numpy()
        blur_np = blurred[0].permute(0, 2, 3, 1).cpu().numpy()

        export_to_video(vid_np, os.path.join(args.out_dir, f"clean_{i:05d}.mp4"), fps=16)
        export_to_video(blur_np, os.path.join(args.out_dir, f"blur_{i:05d}.mp4"), fps=16)

        save_image(video[0,0].cpu(), os.path.join(args.out_dir, f"clean_{i:05d}_f0.png"))
        save_image(blurred[0,0].cpu(), os.path.join(args.out_dir, f"blur_{i:05d}_f0.png"))

        with open(os.path.join(args.out_dir, f"prompt_{i:05d}.txt"), "w") as f:
            f.write(prompt)

    print(f"Exported {total} validation videos to {args.out_dir}")


if __name__ == "__main__":
    main() 