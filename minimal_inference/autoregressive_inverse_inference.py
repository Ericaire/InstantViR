from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset, InverseProblemLMDBDataset, PredegradedLMDBDataset # Import new dataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os
import time
from torch.utils.data import random_split # Import for splitting dataset
import torch.nn.functional as F

# program start wall-clock for T_init
_program_start_time = time.time()

# Import video operators for inverse problems
from causvid.models.wan.video_operators import (
    temporal_blur_latent, add_noise_latent
)
from causvid.models.wan.wan_wrapper import WanVAEWrapper

# Extend InferencePipeline to support inverse problems
class InverseProblemPipeline(InferencePipeline):
    """Extended pipeline that supports inverse problem solving"""
    def __init__(self, args, device):
        super().__init__(args, device)
        self.device = device # Explicitly save the device
        self.args = args  # Store args for accessing inverse_problem_type
        # Infer dtype from a model parameter, which is the most reliable way
        self.dtype = next(self.generator.parameters()).dtype
        # Ensure frame_seq_length: honor parent dynamic value; optionally override if provided in args
        if hasattr(args, "frame_sequence_length"):
            self.frame_seq_length = int(args.frame_sequence_length)
        # timing container for last inference call
        self.last_timings = {}

    @staticmethod
    def _num_bytes(x: torch.Tensor) -> int:
        return int(x.numel() * x.element_size())

    @staticmethod
    def _format_bytes(n: int) -> str:
        n_f = float(n)
        for u in ["B", "KB", "MB", "GB", "TB", "PB"]:
            if n_f < 1024.0:
                return f"{n_f:.2f}{u}"
            n_f /= 1024.0
        return f"{n_f:.2f}EB"

    def _summarize_kv_cache_bytes(self) -> dict:
        kv_bytes = 0
        if getattr(self, "kv_cache1", None) is not None:
            for layer in self.kv_cache1:
                if isinstance(layer, dict) and ("k" in layer) and ("v" in layer):
                    kv_bytes += self._num_bytes(layer["k"]) + self._num_bytes(layer["v"])

        xattn_bytes = 0
        if getattr(self, "crossattn_cache", None) is not None:
            for layer in self.crossattn_cache:
                if isinstance(layer, dict) and ("k" in layer) and ("v" in layer):
                    xattn_bytes += self._num_bytes(layer["k"]) + self._num_bytes(layer["v"])

        return {
            "kv_bytes": int(kv_bytes),
            "crossattn_bytes": int(xattn_bytes),
            "total_bytes": int(kv_bytes + xattn_bytes),
        }

    def _print_cuda_mem(self, tag: str):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        alloc = int(torch.cuda.memory_allocated())
        reserv = int(torch.cuda.memory_reserved())
        peak_alloc = int(torch.cuda.max_memory_allocated())
        peak_reserv = int(torch.cuda.max_memory_reserved())
        print(
            f"[MEM][{tag}] "
            f"alloc={self._format_bytes(alloc)} reserv={self._format_bytes(reserv)} | "
            f"peak_alloc={self._format_bytes(peak_alloc)} peak_reserv={self._format_bytes(peak_reserv)}"
        )
    
    def inference_inverse(self, degraded_observation: torch.Tensor, text_prompts: list, 
                         return_latents: bool = False) -> torch.Tensor:
        """
        Perform inverse problem inference using degraded observation as input.
        Uses the same block-wise processing and KV caching as standard inference.
        """
        # Ensure input tensor has the same dtype as the model (important when pipeline is cast to bf16/fp16)
        model_dtype = next(self.generator.parameters()).dtype
        self.dtype = model_dtype
        degraded_observation = degraded_observation.to(model_dtype)

        batch_size, num_frames, num_channels, height, width = degraded_observation.shape
        # Reset peak memory stats right before we start the actual inference work.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self._print_cuda_mem("start")
        
        # Get text embeddings (timed)
        torch.cuda.synchronize()
        _t_text_s = time.time()
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        torch.cuda.synchronize()
        _t_text = time.time() - _t_text_s
        
        # Initialize output tensor
        output = torch.zeros_like(degraded_observation)
        
        # Initialize KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=degraded_observation.dtype,
                device=degraded_observation.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=degraded_observation.dtype,
                device=degraded_observation.device
            )
        cache_bytes = self._summarize_kv_cache_bytes()
        print(
            f"[KV] kv_cache={self._format_bytes(cache_bytes['kv_bytes'])} "
            f"crossattn_cache={self._format_bytes(cache_bytes['crossattn_bytes'])} "
            f"total={self._format_bytes(cache_bytes['total_bytes'])}"
        )
        self._print_cuda_mem("after_cache_init")
        
        # Process block by block
        num_blocks = (num_frames + self.num_frame_per_block - 1) // self.num_frame_per_block
        
        _t_core = 0.0
        for block_idx in range(num_blocks):
            start_frame_idx = block_idx * self.num_frame_per_block
            end_frame_idx = min(start_frame_idx + self.num_frame_per_block, num_frames)
            num_frames_in_block = end_frame_idx - start_frame_idx
            
            # Get current block of degraded observation
            current_observation = degraded_observation[:, start_frame_idx:end_frame_idx]
            
            # For inverse problems, use appropriate timestep based on training
            # Match the timestep used during training for consistency
            if hasattr(self, 'args') and hasattr(self.args, 'inverse_problem_type'):
                if self.args.inverse_problem_type == "spatial_blur":
                    timestep_value = 522  # Match training timestep
                else:
                    timestep_value = 522  # Default for other problems
            else:
                timestep_value = 522  # Use 522 for spatial blur inference
            
            timestep = torch.full(
                [batch_size, num_frames_in_block], 
                timestep_value,
                device=degraded_observation.device, 
                dtype=torch.long
            )
            
            # Run generator with KV cache (timed)
            torch.cuda.synchronize()
            _ts = time.time()
            denoised_pred = self.generator(
                noisy_image_or_video=current_observation,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=start_frame_idx * self.frame_seq_length,
                current_end=end_frame_idx * self.frame_seq_length
            )
            torch.cuda.synchronize()
            _t_core += (time.time() - _ts)
            
            output[:, start_frame_idx:end_frame_idx] = denoised_pred
            
            # Update KV cache positions
            if hasattr(self.generator.model, 'update_kv_cache_position'):
                self.generator.model.update_kv_cache_position(
                    self.kv_cache1, self.kv_cache2, 
                    block_idx, num_frames_in_block
                )
        self._print_cuda_mem("after_core")
        
        if return_latents:
            return output
        
        # Decode to video (timed)
        torch.cuda.synchronize()
        _t_dec_s = time.time()
        video = self.vae.decode_to_pixel(output)
        torch.cuda.synchronize()
        _t_decode = time.time() - _t_dec_s
        self._print_cuda_mem("after_decode")
        
        # Normalize to [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        # Save timings (encode is 0 when using pre-degraded latents)
        self.last_timings = {
            't_encode': 0.0,
            't_text': _t_text,
            't_core': _t_core,
            't_decode': _t_decode,
            'num_frames': int(video.shape[1]),
            'height': int(video.shape[-2]),
            'width': int(video.shape[-1])
        }

        return video

# Main script
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)
# Add inverse problem specific arguments
parser.add_argument("--inverse_problem", action="store_true", help="Enable inverse problem mode")
parser.add_argument("--blur_kernel_size", type=int, default=7, help="Kernel size for blur")
parser.add_argument("--blur_sigma", type=float, default=1.5, help="Sigma for spatial blur")
parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level for degradation")
# Add argument for testing from dataset
parser.add_argument("--test_video_index", type=int, default=None, help="Index of the video in the VALIDATION dataset to test.")
parser.add_argument("--train_video_index", type=int, default=None, help="Index of the video in the TRAINING dataset to test.")
parser.add_argument("--data_path", type=str, default="data/mixkit_latents_lmdb", help="Path to the LMDB dataset.")
# New arguments for custom prompt and dataset mode
parser.add_argument("--use_predegraded_dataset", action="store_true", help="Flag to use the pre-degraded dataset.")
parser.add_argument("--custom_prompt", type=str, default=None, help="Custom text prompt to use for inference, overriding the one from the dataset.")
# Multistep inverse inference options (Scheme A)
parser.add_argument("--infer_timesteps", type=str, default=None, help="Comma-separated timesteps for multi-step inverse inference, e.g., '1000,757,522'. If None, use single step (522).")
parser.add_argument("--num_inverse_steps", type=int, default=None, help="If set and --infer_timesteps is None, take the first N from config.denoising_step_list (excluding 0) in descending order.")
parser.add_argument("--chain_infer", action="store_true", help="Enable chain-style multi-step inference: feed previous step's output as next step's input.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for dataset-based testing (loads consecutive indices starting at the given index).")
parser.add_argument(
    "--infer_dtype",
    type=str,
    default="fp32",
    choices=["fp32", "bf16", "fp16"],
    help="Inference dtype for the whole pipeline (affects KV cache and activations). Default fp32 for reproducibility."
)


args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

# Use extended pipeline for inverse problems
pipeline = InverseProblemPipeline(config, device="cuda")
# pipeline.to(device="cuda", dtype=torch.bfloat16) # This will be handled by the float32 cast below

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu", weights_only=False)[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

# Set inference dtype
_dtype_map = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_infer_dtype = _dtype_map.get(getattr(args, "infer_dtype", "fp32"), torch.float32)
pipeline.to(device="cuda", dtype=_infer_dtype)
# Keep pipeline.dtype consistent with actual model dtype (used for input casts + KV cache allocation).
pipeline.dtype = next(pipeline.generator.parameters()).dtype
print(f"[Infer] infer_dtype={getattr(args, 'infer_dtype', 'fp32')} (torch dtype={pipeline.dtype})")

os.makedirs(args.output_folder, exist_ok=True)

# --- Main logic switch ---

if (args.test_video_index is not None) or (args.train_video_index is not None):
    # --- Mode 1: Test a specific video from the dataset (train or val) ---
    if (args.test_video_index is not None) and (args.train_video_index is not None):
        raise ValueError("Provide only one of --test_video_index or --train_video_index, not both.")
    split_name = "val" if (args.test_video_index is not None) else "train"
    split_index = args.test_video_index if (args.test_video_index is not None) else args.train_video_index
    print(f"--- Testing mode: Loading video index {split_index} from {split_name} set ---")
    
    # --- Dataset Loading and Splitting ---
    if args.use_predegraded_dataset:
        print("--- Using Pre-degraded LMDB Dataset for inference ---")
        full_dataset = PredegradedLMDBDataset(
            data_path=args.data_path
        )
    else:
        print("--- Using On-the-fly Degradation Dataset for inference ---")
        degradation_params = {
            "blur_kernel_size": args.blur_kernel_size,
            "blur_sigma": args.blur_sigma,
            "noise_level": args.noise_level
        }
        full_dataset = InverseProblemLMDBDataset(
            data_path=args.data_path,
            inverse_problem_type=config.inverse_problem_type,
            degradation_params=degradation_params
        )

    # Split dataset into training and validation sets (90/10 split)
    # Use the same seed as in training to get the exact same split
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # --- Build a batch of consecutive samples ---
    if split_name == "val":
        active_dataset = val_dataset
        print(f"Loaded validation set with {len(val_dataset)} samples.")
    else:
        active_dataset = train_dataset
        print(f"Loaded training set with {len(train_dataset)} samples.")

    if split_index >= len(active_dataset):
        raise ValueError(("test_video_index " if split_name=="val" else "train_video_index ") + f"{split_index} is out of bounds for {split_name} set with length {len(active_dataset)}")

    # Determine the actual batch indices (truncate at dataset end)
    max_bsz = max(1, int(args.batch_size))
    indices_to_load = list(range(split_index, min(split_index + max_bsz, len(active_dataset))))
    if len(indices_to_load) < max_bsz:
        print(f"[Warn] Requested batch_size={max_bsz} but only {len(indices_to_load)} samples available from index {split_index}. Using smaller batch.")
    print(f"--- Loading indices: {indices_to_load} ---")

    samples = [active_dataset[i] for i in indices_to_load]
    # Collate tensors to batch
    clean_latent = torch.stack([s["clean_latent"] for s in samples], dim=0).to(pipeline.device, dtype=pipeline.dtype)
    degraded_observation = torch.stack([s["degraded_observation"] for s in samples], dim=0).to(pipeline.device, dtype=pipeline.dtype)
    # Keep a copy of LR measurement for visualization/export
    degraded_observation_lr = degraded_observation

    # For SR×4: upsample LR latent measurement to HR latent grid as generator input; reset caches and set frame_seq_length
    if getattr(config, "inverse_problem_type", None) == "super_resolution":
        # Use the actual GT latent spatial size from the dataset as HR target.
        # This avoids mismatches when config.image_or_video_shape differs from the pre-degraded dataset resolution.
        hr_h = int(clean_latent.shape[-2])
        hr_w = int(clean_latent.shape[-1])
        if degraded_observation.shape[-2] != hr_h or degraded_observation.shape[-1] != hr_w:
            degraded_observation = F.interpolate(
                degraded_observation.flatten(0, 1),
                size=(hr_h, hr_w),
                mode='bilinear', align_corners=False
            ).unflatten(0, degraded_observation.shape[:2])
        patch_size = getattr(pipeline.generator.model, "patch_size", (1, 2, 2))
        if len(patch_size) == 3:
            p_h, p_w = patch_size[1], patch_size[2]
        else:
            p_h, p_w = patch_size[0], patch_size[1]
        pipeline.frame_seq_length = (hr_h * hr_w) // (p_h * p_w)
        # Reinit caches to match HR token length
        pipeline.kv_cache1 = None
        pipeline.kv_cache2 = None
        pipeline.crossattn_cache = None
    
    # --- Prompt Handling ---
    if args.custom_prompt:
        print(f"--- Using custom prompt (same for batch): '{args.custom_prompt}' ---")
        prompts = [args.custom_prompt for _ in range(clean_latent.shape[0])]
    else:
        prompts = [s["prompts"] for s in samples]
        print(f"--- Using prompts from dataset (batch={len(prompts)}), example[0]: '{prompts[0]}' ---")

    # --- Run Inference and Time it ---
    # Mark init end to compute T_init (from program start to inference start)
    torch.cuda.synchronize()
    init_end_time = time.time()
    start_time = time.time()

    # Multistep inverse inference (Scheme A)
    infer_t_list = None
    if args.infer_timesteps:
        infer_t_list = [int(x) for x in args.infer_timesteps.split(',') if x.strip()]
    elif args.num_inverse_steps and args.num_inverse_steps > 1:
        # Take from config.denoising_step_list excluding 0, in descending order
        t_list = [int(t) for t in config.denoising_step_list if int(t) != 0]
        t_list_sorted = sorted(t_list, reverse=True)
        infer_t_list = t_list_sorted[:args.num_inverse_steps]

    if infer_t_list is None or len(infer_t_list) == 0:
        # Single step (default t=522)
        reconstructed_video_tensor = pipeline.inference_inverse(
            degraded_observation=degraded_observation,
            text_prompts=prompts,
            return_latents=False
        )
    else:
        # Run inference multiple times.
        # If chain_infer=False: each round uses the same degraded_observation (Scheme A)
        # If chain_infer=True:  each round uses previous output as the next input (Scheme B inference)
        current_input = degraded_observation
        last_latent = None
        _t_text_total = 0.0
        _t_core_total = 0.0
        for t_val in infer_t_list:
            # Reinit KV/cross-attn caches for inference path (use causal KV route)
            pipeline.kv_cache1 = None
            pipeline.kv_cache2 = None
            pipeline.crossattn_cache = None
            # Initialize caches (same as pipeline.inference_inverse)
            bsz = current_input.shape[0]
            if pipeline.kv_cache1 is None:
                pipeline._initialize_kv_cache(
                    batch_size=bsz,
                    dtype=current_input.dtype,
                    device=current_input.device
                )
                pipeline._initialize_crossattn_cache(
                    batch_size=bsz,
                    dtype=current_input.dtype,
                    device=current_input.device
                )

            batch_size, num_frames, _, _, _ = current_input.shape
            num_blocks = (num_frames + pipeline.num_frame_per_block - 1) // pipeline.num_frame_per_block
            torch.cuda.synchronize()
            __t_ts = time.time()
            conditional_dict = pipeline.text_encoder(text_prompts=prompts)
            torch.cuda.synchronize()
            _t_text_total += (time.time() - __t_ts)
            output = torch.zeros_like(current_input)
            for block_idx in range(num_blocks):
                s = block_idx * pipeline.num_frame_per_block
                e = min(s + pipeline.num_frame_per_block, num_frames)
                cur = current_input[:, s:e]
                timestep = torch.full([batch_size, e - s], t_val, device=current_input.device, dtype=torch.long)
                torch.cuda.synchronize()
                __t_cs = time.time()
                denoised_pred = pipeline.generator(
                    noisy_image_or_video=cur,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=s * pipeline.frame_seq_length,
                    current_end=e * pipeline.frame_seq_length
                )
                torch.cuda.synchronize()
                _t_core_total += (time.time() - __t_cs)
                output[:, s:e] = denoised_pred
                if hasattr(pipeline.generator.model, 'update_kv_cache_position'):
                    pipeline.generator.model.update_kv_cache_position(
                        pipeline.kv_cache1, pipeline.kv_cache2, block_idx, e - s
                    )
            last_latent = output
            if args.chain_infer:
                current_input = output  # feed last output into next round
            # else keep current_input as original degraded_observation (Scheme A)

        # decode timing
        torch.cuda.synchronize()
        __t_ds = time.time()
        reconstructed_video_tensor = pipeline.vae.decode_to_pixel(last_latent)
        torch.cuda.synchronize()
        _t_decode = time.time() - __t_ds
        reconstructed_video_tensor = ((reconstructed_video_tensor * 0.5) + 0.5).clamp(0.0, 1.0)
        pipeline.last_timings = {
            't_encode': 0.0,
            't_text': _t_text_total,
            't_core': _t_core_total,
            't_decode': _t_decode,
        }
    
    torch.cuda.synchronize() # Wait for inference to complete
    end_time = time.time()
    
    duration = end_time - start_time
    B = reconstructed_video_tensor.shape[0]
    num_frames = reconstructed_video_tensor.shape[1]
    fps = num_frames / duration  # per-video effective FPS over the measured window
    fps_total = (B * num_frames) / duration  # throughput FPS across the whole batch

    # Additional detailed timings
    T_init = init_end_time - globals().get('_program_start_time', init_end_time)
    t_encode = getattr(pipeline, 'last_timings', {}).get('t_encode', 0.0)
    t_text = getattr(pipeline, 'last_timings', {}).get('t_text', float('nan'))
    t_core = getattr(pipeline, 'last_timings', {}).get('t_core', float('nan'))
    t_decode = getattr(pipeline, 'last_timings', {}).get('t_decode', float('nan'))
    fps_core = num_frames / t_core if (t_core and t_core > 0) else float('nan')
    fps_core_total = (B * num_frames) / t_core if (t_core and t_core > 0) else float('nan')
    fps_e2e_steady = num_frames / (t_encode + t_core + t_decode) if (t_core and t_decode) else float('nan')
    fps_e2e_steady_total = (B * num_frames) / (t_encode + t_core + t_decode) if (t_core and t_decode) else float('nan')

    print(f"--- Inference complete ---")
    print(f"  - Time taken (measure window): {duration:.3f} s")
    print(f"  - Batch size: {B}")
    print(f"  - Frames per video: {num_frames}")
    print(f"  - Inference speed (per-video window): {fps:.2f} FPS")
    print(f"  - Inference throughput (batch total): {fps_total:.2f} FPS")
    print(f"  - T_init (program start -> infer start): {T_init:.3f} s")
    print(f"  - t_encode: {t_encode:.3f} s (pre-degraded latents => ~0)")
    print(f"  - t_text: {t_text:.3f} s")
    print(f"  - t_core: {t_core:.3f} s  | FPS_core(per-video): {fps_core:.2f} | FPS_core(total): {fps_core_total:.2f}")
    print(f"  - t_decode: {t_decode:.3f} s  | FPS_e2e(no I/O, per-video): {fps_e2e_steady:.2f} | FPS_e2e(no I/O, total): {fps_e2e_steady_total:.2f}")

    # --- Save all versions for comparison (for each sample in the batch) ---
    # Decode LR/decode clean in batch once to avoid extra compute
    degraded_px_lr = pipeline.vae.decode_to_pixel(degraded_observation_lr)
    degraded_px_lr = ((degraded_px_lr * 0.5) + 0.5).clamp(0.0, 1.0)
    clean_video_all = pipeline.vae.decode_to_pixel(clean_latent)
    clean_video_all = ((clean_video_all * 0.5) + 0.5).clamp(0.0, 1.0)
    target_h_px = int(reconstructed_video_tensor.shape[-2])
    target_w_px = int(reconstructed_video_tensor.shape[-1])
    btchw = degraded_px_lr.flatten(0, 1)  # [B*T, C, H_lr, W_lr]
    degraded_px_up = F.interpolate(btchw, size=(target_h_px, target_w_px), mode='bicubic', align_corners=False).unflatten(0, degraded_px_lr.shape[:2])

    for bi, sample_idx in enumerate(indices_to_load):
        # 1) reconstructed
        # numpy doesn't support bf16; cast on CPU to avoid extra GPU memory
        reconstructed_video = reconstructed_video_tensor[bi].permute(0, 2, 3, 1).cpu().float().numpy()
        export_to_video(
            reconstructed_video,
            os.path.join(args.output_folder, f"reconstructed_{split_name}_{sample_idx:03d}.mp4"),
            fps=16
        )
        # 2) degraded upsampled and LR
        degraded_video_up = degraded_px_up[bi].permute(0, 2, 3, 1).cpu().float().numpy()
        export_to_video(
            degraded_video_up,
            os.path.join(args.output_folder, f"degraded_{split_name}_{sample_idx:03d}_upx4.mp4"),
            fps=16
        )
        degraded_video_lr = degraded_px_lr[bi].permute(0, 2, 3, 1).cpu().float().numpy()
        export_to_video(
            degraded_video_lr,
            os.path.join(args.output_folder, f"degraded_{split_name}_{sample_idx:03d}_lr.mp4"),
            fps=16
        )
        # 3) clean
        clean_video = clean_video_all[bi].permute(0, 2, 3, 1).cpu().float().numpy()
        export_to_video(
            clean_video,
            os.path.join(args.output_folder, f"original_{split_name}_{sample_idx:03d}.mp4"),
            fps=16
        )
    
    print(f"--- Videos saved to {args.output_folder} ---")

else:
    # --- Mode 2: Original self-generation loop (remains unchanged) ---
    print("--- Debugging mode: This mode is for prompt-based generation from noise and is not affected by dataset changes. ---")
    dataset = TextDataset(args.prompt_file_path)
    
    for prompt_index in tqdm(range(len(dataset))):
        prompts = [dataset[prompt_index]]
        
        if args.inverse_problem:
            # For testing: generate a clean video and degrade it
            sampled_noise = torch.randn(
                [1, 21, 16, 60, 104], device="cuda", dtype=torch.float32
            )
            
            # Generate clean latent
            clean_video, clean_latent = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True
            )
            
            # Apply degradation based on config
            if config.inverse_problem_type == "spatial_blur":
                from causvid.models.wan.video_operators import spatial_blur_latent
                degraded_observation = spatial_blur_latent(clean_latent, kernel_size_s=args.blur_kernel_size, sigma_s=args.blur_sigma)
            else: # Default to temporal
                degraded_observation = temporal_blur_latent(clean_latent, kernel_size_t=args.blur_kernel_size)
            
            degraded_observation = add_noise_latent(
                degraded_observation, noise_level=args.noise_level
            )
            
            # Use inverse problem inference with proper KV caching
            video = pipeline.inference_inverse(
                degraded_observation=degraded_observation,
                text_prompts=prompts,
                return_latents=False
            )[0].permute(0, 2, 3, 1).cpu().numpy()
            
            # Also save degraded video for comparison
            degraded_video = pipeline.vae.decode_to_pixel(degraded_observation)
            degraded_video = ((degraded_video * 0.5) + 0.5).clamp(0.0, 1.0)
            degraded_video = degraded_video[0].permute(0, 2, 3, 1).cpu().numpy()
            
            export_to_video(
                degraded_video, 
                os.path.join(args.output_folder, f"degraded_{prompt_index:03d}.mp4"), 
                fps=16
            )
        else:
            # Standard generation mode
            sampled_noise = torch.randn(
                [1, 21, 16, 60, 104], device="cuda", dtype=torch.float32
            )
            
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts
            )[0].permute(0, 2, 3, 1).cpu().numpy()

        export_to_video(
            video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)