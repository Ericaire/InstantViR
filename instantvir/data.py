from instantvir.ode_data.create_lmdb_iterative import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
from instantvir.models.wan.video_operators import (
    temporal_blur_latent, add_noise_latent, random_mask_latent, downsample_latent,
    spatial_blur_latent, generate_inpainting_mask
)


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ODERegressionDataset(Dataset):
    def __init__(self, data_path, max_pair=int(1e8)):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair

    def __len__(self):
        return min(len(self.data_dict['prompts']), self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        return {
            "prompts": self.data_dict['prompts'][idx],
            "ode_latent": self.data_dict['latents'][idx].squeeze(0),
        }


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            # Case 1: standard MixKit latent LMDB (no ODE dimension)
            # Stored as [T, C, H, W] → add a singleton "ODE step" axis at dim=0:
            # [1, T, C, H, W], so that
            #   - ODE training (which expects [S, T, C, H, W]) can still treat S=1
            #   - DMD training with backward_simulation=False can take
            #     clean_latent = batch["ode_latent"][:, -1]  → [B, T, C, H, W]
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class InverseProblemLMDBDataset(Dataset):
    """Dataset for video inverse problems. Loads clean latents and applies degradation."""
    
    def __init__(self, data_path: str, inverse_problem_type: str, 
                 degradation_params: dict = None, max_samples: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)
        
        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_samples = max_samples
        self.inverse_problem_type = inverse_problem_type
        self.degradation_params = degradation_params or {}
        
        # Setup degradation operator
        self.setup_degradation_operator()
    
    def setup_degradation_operator(self):
        """Setup the degradation operator based on problem type"""
        if self.inverse_problem_type == "gaussian_blur":
            kernel_size = self.degradation_params.get("blur_kernel_size", 7)
            noise_level = self.degradation_params.get("noise_level", 0.05)
            self.degradation_fn = lambda x: add_noise_latent(
                temporal_blur_latent(x, kernel_size_t=kernel_size),
                noise_level=noise_level
            )
        elif self.inverse_problem_type == "spatial_blur":
            kernel_size = self.degradation_params.get("blur_kernel_size", 7)
            sigma = self.degradation_params.get("blur_sigma", 1.5)
            noise_level = self.degradation_params.get("noise_level", 0.05)
            self.degradation_fn = lambda x: add_noise_latent(
                spatial_blur_latent(x, kernel_size_s=kernel_size, sigma_s=sigma),
                noise_level=noise_level
            )
        elif self.inverse_problem_type == "super_resolution":
            scale_factor = self.degradation_params.get("scale_factor", 0.25)
            self.degradation_fn = lambda x: downsample_latent(x, scale_factor=scale_factor)
        elif self.inverse_problem_type == "inpainting":
            mask_type = self.degradation_params.get("inpainting_mask_type", "random")
            box_size = self.degradation_params.get("inpainting_box_size", [30, 52])
            self.degradation_fn = lambda x: x * generate_inpainting_mask(x, mask_type, box_size)
        else:
            raise ValueError(f"Unknown inverse problem type: {self.inverse_problem_type}")
    
    def __len__(self):
        return min(self.latents_shape[0], self.max_samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            - prompts: Text prompt string
            - clean_latent: Clean latent tensor [T, C, H, W]
            - degraded_observation: Degraded latent tensor [T, C, H, W]
        """
        # Load clean latent
        clean_latent = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )
        
        # ODE dataset has shape [num_denoising_steps, T, C, H, W]
        # We only need the clean latent (last step)
        if len(clean_latent.shape) == 5:
            clean_latent = clean_latent[-1]  # Take the last (cleanest) step
        elif len(clean_latent.shape) == 4:
            # If shape is [T, C, H, W], it's already what we need
            pass
        elif len(clean_latent.shape) == 1:
            # Reshape flattened array from LMDB
            # This is specific to the pre-processed latents in mixkit_latents_lmdb
            num_elements = 21 * 16 * 60 * 104
            if clean_latent.size == num_elements:
                clean_latent = clean_latent.reshape(21, 16, 60, 104)
            else:
                raise ValueError(
                    f"Flattened latent has incorrect size: {clean_latent.size}. "
                    f"Expected {num_elements}."
                )
        else:
            raise ValueError(f"Unexpected clean_latent shape: {clean_latent.shape}")
        
        clean_latent = torch.tensor(clean_latent)
        
        # Apply degradation
        # Add batch dimension for operators
        clean_latent_batch = clean_latent.unsqueeze(0)  # [1, T, C, H, W]
        degraded_observation = self.degradation_fn(clean_latent_batch)
        degraded_observation = degraded_observation.squeeze(0)  # [T, C, H, W]
        
        # Load prompt
        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
        
        return {
            "prompts": prompts,
            "clean_latent": clean_latent,
            "degraded_observation": degraded_observation
        }

class PredegradedLMDBDataset(Dataset):
    """Dataset for loading pre-degraded latents from an LMDB database."""
    
    def __init__(self, data_path: str, max_samples: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        def _parse_shape_str(s: str):
            if s is None:
                return None
            s = s.strip()
            # allow both comma and space delimiters
            parts = [p for p in s.replace(',', ' ').split(' ') if p]
            return tuple(map(int, parts))

        # Get shapes of the stored arrays (兼容单Mask与双Mask)
        with self.env.begin() as txn:
            clean_raw = txn.get("clean_latent_shape".encode())
            degraded_raw = txn.get("degraded_latent_shape".encode())
            degraded_fg_raw = txn.get("degraded_latent_fg_shape".encode())
            degraded_bg_raw = txn.get("degraded_latent_bg_shape".encode())
            self.clean_shape = _parse_shape_str(clean_raw.decode()) if clean_raw is not None else None
            # 单Mask（旧格式）
            self.degraded_shape = _parse_shape_str(degraded_raw.decode()) if degraded_raw is not None else None
            # 双Mask（新格式）
            self.degraded_fg_shape = _parse_shape_str(degraded_fg_raw.decode()) if degraded_fg_raw is not None else None
            self.degraded_bg_shape = _parse_shape_str(degraded_bg_raw.decode()) if degraded_bg_raw is not None else None
            # 判定格式
            self.is_dual_mask = (self.degraded_fg_shape is not None) and (self.degraded_bg_shape is not None)
            if (self.degraded_shape is None) and (not self.is_dual_mask):
                raise ValueError("LMDB缺少 degraded_latent_shape 或 degraded_latent_fg/bg_shape 任何一种形状键，无法识别为单/双Mask数据。")

            mask_shape_raw = txn.get("inpainting_mask_shape".encode())
            self.mask_shape = _parse_shape_str(mask_shape_raw.decode()) if mask_shape_raw is not None else None
        
        self.max_samples = max_samples
    
    def __len__(self):
        # The number of samples is the first element of the shape tuple
        return min(self.clean_shape[0], self.max_samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            - prompts: Text prompt string
            - clean_latent: Clean latent tensor
            - degraded_observation: For 单Mask, a pre-degraded latent tensor
            - degraded_observation_fg/bg: For 双Mask, foreground/background degraded latents
        """
        # Load clean latent
        clean_latent = retrieve_row_from_lmdb(
            self.env,
            "clean_latent", np.float16, idx, shape=self.clean_shape[1:]
        )
        
        sample = {
            "clean_latent": torch.tensor(clean_latent)
        }

        if self.is_dual_mask:
            # 双Mask：读取 fg/bg 两路
            degraded_latent_fg = retrieve_row_from_lmdb(
                self.env,
                "degraded_latent_fg", np.float16, idx, shape=self.degraded_fg_shape[1:]
            )
            degraded_latent_bg = retrieve_row_from_lmdb(
                self.env,
                "degraded_latent_bg", np.float16, idx, shape=self.degraded_bg_shape[1:]
            )
            sample["degraded_observation_fg"] = torch.tensor(degraded_latent_fg)
            sample["degraded_observation_bg"] = torch.tensor(degraded_latent_bg)
            # 兼容：提供一个合成观测，方便未适配代码至少能跑（fg+bg≈全视频）
            try:
                sample["degraded_observation"] = sample["degraded_observation_fg"] + sample["degraded_observation_bg"]
            except Exception:
                pass
        else:
            # 单Mask：读取旧键
            degraded_latent = retrieve_row_from_lmdb(
                self.env,
                "degraded_latent", np.float16, idx, shape=self.degraded_shape[1:]
            )
            sample["degraded_observation"] = torch.tensor(degraded_latent)  # Return under the key the trainer expects
        
        # Load prompt
        prompts = retrieve_row_from_lmdb(self.env, "prompts", str, idx)
        sample["prompts"] = prompts

        if self.mask_shape is not None:
            mask = retrieve_row_from_lmdb(
                self.env, "inpainting_mask", np.float16, idx, shape=self.mask_shape[1:]
            )
            sample["inpainting_mask"] = torch.tensor(mask)  # [H,W]
        return sample
