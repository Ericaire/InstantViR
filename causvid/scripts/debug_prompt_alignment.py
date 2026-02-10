import argparse
import os
import random
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from causvid.data import InverseProblemLMDBDataset, PredegradedLMDBDataset
from causvid.models.wan.causal_inference import InferencePipeline


def decode_first_frame(vae_wrapper, latent_btchw: torch.Tensor) -> torch.Tensor:
    # latent_btchw: [1, T, C, H, W]
    with torch.no_grad():
        video = vae_wrapper.decode_to_pixel(latent_btchw)
        video = (video * 0.5 + 0.5).clamp(0, 1)  # to [0,1]
        frame0 = video[0, 0]  # [C, H, W]
        return frame0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--use_predegraded_dataset", action="store_true")
    parser.add_argument("--indices", type=str, default="0,10,100")
    parser.add_argument("--out_dir", type=str, default="debug_prompt_alignment")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = OmegaConf.load(args.config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = InferencePipeline(config, device=device)
    try:
        pipeline.vae.model.to(device)
    except Exception:
        pass

    # Load dataset
    if args.use_predegraded_dataset:
        dataset = PredegradedLMDBDataset(args.data_path)
    else:
        dataset = InverseProblemLMDBDataset(
            data_path=args.data_path,
            inverse_problem_type=getattr(config, "inverse_problem_type", "spatial_blur"),
            degradation_params={}
        )

    # Parse indices
    idx_list = [int(x) for x in args.indices.split(",") if x.strip()]

    print(f"Dataset length: {len(dataset)}")

    for idx in idx_list:
        if idx >= len(dataset):
            print(f"Index {idx} out of range")
            continue
        sample = dataset[idx]
        prompt = sample["prompts"]
        print(f"[idx={idx}] prompt: {prompt}")

        clean_latent = sample["clean_latent"].unsqueeze(0).to(device=device, dtype=torch.float32)
        clean_frame0 = decode_first_frame(pipeline.vae, clean_latent)
        save_image(clean_frame0, os.path.join(args.out_dir, f"idx{idx:05d}_clean_frame0.png"))

        if "degraded_observation" in sample:
            degraded = sample["degraded_observation"].unsqueeze(0).to(device=device, dtype=torch.float32)
            degraded_frame0 = decode_first_frame(pipeline.vae, degraded)
            save_image(degraded_frame0, os.path.join(args.out_dir, f"idx{idx:05d}_degraded_frame0.png"))

    print(f"Saved frames to {args.out_dir}")


if __name__ == "__main__":
    main() 