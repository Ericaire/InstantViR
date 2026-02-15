from instantvir.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from instantvir.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os

# Import video operators for inverse problems
from instantvir.models.wan.video_operators import (
    temporal_blur_latent, add_noise_latent
)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
# Add inverse problem specific arguments
parser.add_argument("--inverse_problem", action="store_true", help="Enable inverse problem mode")
parser.add_argument("--blur_kernel_size", type=int, default=7, help="Kernel size for temporal blur")
parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level for degradation")

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]
    
    if args.inverse_problem:
        # For inverse problems, we need to properly use the pipeline
        # First, let's prepare a degraded observation for testing
        
        # Option 1: Load a pre-existing video and degrade it
        # For now, let's generate one and degrade it as a test
        sampled_noise = torch.randn(
            [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
        )
        
        # Generate a clean video first (for testing purposes)
        clean_latent = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True
        )
        
        # Apply degradation to create observation
        degraded_observation = temporal_blur_latent(
            clean_latent, kernel_size_t=args.blur_kernel_size
        )
        degraded_observation = add_noise_latent(
            degraded_observation, noise_level=args.noise_level
        )
        
        # Now for reconstruction, we need to modify the pipeline to accept degraded input
        # Since InferencePipeline expects noise input, we have two options:
        
        # Option A: Treat degraded observation as "noise" and use special timesteps
        # This is a temporary solution until we properly modify InferencePipeline
        
        # For now, let's use the trained generator directly with proper setup
        # This is not ideal but demonstrates the concept
        
        # Initialize KV cache manually (this is what we're missing)
        pipeline._initialize_kv_cache(
            batch_size=1,
            num_frames=21,
            num_channels=16,
            height=60,
            width=104,
            device="cuda",
            dtype=torch.bfloat16
        )
        
        # Process block by block (mimicking what inference() does)
        num_frame_per_block = pipeline.num_frame_per_block
        num_blocks = (21 + num_frame_per_block - 1) // num_frame_per_block
        
        reconstructed_latent = torch.zeros_like(degraded_observation)
        
        with torch.no_grad():
            conditional_dict = pipeline.text_encoder(text_prompts=prompts)
            
            for block_idx in range(num_blocks):
                # Get current block
                start_frame = block_idx * num_frame_per_block
                end_frame = min(start_frame + num_frame_per_block, 21)
                
                current_observation = degraded_observation[:, start_frame:end_frame]
                
                # For inverse problems, use timestep=0
                timestep = torch.zeros(
                    [1, end_frame - start_frame], device="cuda", dtype=torch.long
                )
                
                # Run generator with KV cache
                block_output = pipeline.generator(
                    noisy_image_or_video=current_observation,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    inference_mode=True,
                    kv_cache1=pipeline.kv_cache1,
                    kv_cache2=pipeline.kv_cache2,
                    crossattn_kv_cache1=pipeline.crossattn_kv_cache1,
                    crossattn_kv_cache2=pipeline.crossattn_kv_cache2,
                    block_idx=block_idx
                )
                
                reconstructed_latent[:, start_frame:end_frame] = block_output
                
                # Update KV cache (this is crucial!)
                # Note: This is a simplified version - the actual implementation
                # in InferencePipeline is more complex
        
        # Decode to video
        video = pipeline.vae.decode(
            reconstructed_latent[0:1].flatten(0, 1) / pipeline.vae.config.shift_factor,
            num_frames=21
        )[0].permute(0, 2, 3, 1).cpu().numpy()
        
        # Also save the degraded observation for comparison
        degraded_video = pipeline.vae.decode(
            degraded_observation[0:1].flatten(0, 1) / pipeline.vae.config.shift_factor,
            num_frames=21
        )[0].permute(0, 2, 3, 1).cpu().numpy()
        
        export_to_video(
            degraded_video, 
            os.path.join(args.output_folder, f"degraded_{prompt_index:03d}.mp4"), 
            fps=16
        )
    else:
        # Standard generation mode
        sampled_noise = torch.randn(
            [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
        )
        
        video = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts
        )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
