import os
import argparse
import lmdb
import numpy as np
import torch
from tqdm import tqdm

# Local imports
from causvid.ode_data.create_lmdb_iterative import (
    get_array_shape_from_lmdb,
    retrieve_row_from_lmdb,
    store_arrays_to_lmdb,
)
from causvid.models.wan.wan_wrapper import WanVAEWrapper
from causvid.models.leanvae_wrapper import LeanVAEWrapper


def convert_sample_wan_to_lean(
    wan_vae: WanVAEWrapper,
    lean_vae: LeanVAEWrapper,
    ode_latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert one video's ODE trajectories from WAN latent space to LeanVAE latent space.
    Args:
        ode_latents: [S, T, C, H, W] OR [T, C, H, W] float16/float32 tensor
    Returns:
        lean_ode_latents: same rank as input, channel converted to 16, in LeanVAE latent space
    """
    with torch.no_grad():
        if ode_latents.ndim == 5:
            S, T, C, H, W = ode_latents.shape
            lean_list = []
            for s in range(S):
                wan_lat = ode_latents[s:s+1].to(device=device, dtype=dtype)  # [1, T, C, H, W]
                pixels = wan_vae.decode_video(wan_lat)                       # [1, T_pix, 3, H, W]
                lean_lat = lean_vae.encode_video(pixels)                     # [1, T_lean, 16, H/8, W/8]
                lean_list.append(lean_lat.to(torch.float16).cpu())
            lean_ode = torch.cat(lean_list, dim=0)                           # [S, T_lean, 16, H/8, W/8]
            return lean_ode
        elif ode_latents.ndim == 4:
            T, C, H, W = ode_latents.shape
            wan_lat = ode_latents.unsqueeze(0).to(device=device, dtype=dtype) # [1, T, C, H, W]
            pixels = wan_vae.decode_video(wan_lat)                            # [1, T_pix, 3, H, W]
            lean_lat = lean_vae.encode_video(pixels)                          # [1, T_lean, 16, H/8, W/8]
            return lean_lat.squeeze(0).to(torch.float16).cpu()                # [T_lean, 16, H/8, W/8]
        else:
            raise ValueError(f"Unexpected ode_latents ndim: {ode_latents.ndim}")


def main():
    parser = argparse.ArgumentParser(description="Convert ODE LMDB from WAN latent to LeanVAE latent")
    parser.add_argument("--input_lmdb", type=str, required=True, help="Path to input ODE LMDB (WAN latent)")
    parser.add_argument("--output_lmdb", type=str, required=True, help="Path to output ODE LMDB (Lean latent)")
    parser.add_argument("--leanvae_ckpt", type=str, default="LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=10**12)
    parser.add_argument("--resume", action="store_true", help="Auto resume from existing output LMDB")
    parser.add_argument("--start_index", type=int, default=None, help="Manually specify starting sample index")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Open input LMDB
    in_env = lmdb.open(args.input_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    latents_shape = get_array_shape_from_lmdb(in_env, "latents")  # (N, S, T, C, H, W)
    num_samples = min(latents_shape[0], args.max_samples)

    # Prepare output LMDB
    # Map size: conservative large value (0.5 TB) to avoid map full; adjust if needed
    out_env = lmdb.open(args.output_lmdb, map_size=512 * 1024**3)

    # Determine start index for resume if requested
    start_idx = 0
    if args.start_index is not None:
        start_idx = int(args.start_index)
        print(f"[Resume] Using manual start_index = {start_idx}")
    elif args.resume:
        # Try read saved shape
        try:
            out_lat_shape = get_array_shape_from_lmdb(out_env, "latents")
            start_idx = int(out_lat_shape[0])
            print(f"[Resume] Found saved latents_shape, start_index = {start_idx}")
        except Exception:
            # Fallback: scan keys to find first missing index from 0..
            with out_env.begin() as txn:
                cur = txn.cursor()
                existing = set()
                for k, _ in cur:
                    if k.startswith(b"latents_") and k.endswith(b"_data"):
                        try:
                            idx = int(k[len(b"latents_"):-len(b"_data")])
                            existing.add(idx)
                        except Exception:
                            pass
                # find first missing index
                i = 0
                while i in existing:
                    i += 1
                start_idx = i
            print(f"[Resume] Scanned keys, start_index = {start_idx}")

    # Init VAEs
    wan_vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)
    lean_vae = LeanVAEWrapper(ckpt_path=args.leanvae_ckpt).to(device=device, dtype=dtype).eval().requires_grad_(False)

    counter = start_idx
    # Convert row by row to save memory
    for idx in tqdm(range(start_idx, num_samples), desc="Converting ODE pairs (WAN -> LeanVAE)"):
        # Read WAN ODE latents and prompt
        ode_lat = retrieve_row_from_lmdb(
            in_env, "latents", np.float16, idx, shape=latents_shape[1:]
        )  # np.float16, shape [S, T, C, H, W]
        prompts = retrieve_row_from_lmdb(in_env, "prompts", str, idx)

        ode_lat_t = torch.tensor(ode_lat, dtype=torch.float16)  # [S, T, C, H, W]
        lean_ode_t = convert_sample_wan_to_lean(
            wan_vae=wan_vae,
            lean_vae=lean_vae,
            ode_latents=ode_lat_t,
            device=device,
            dtype=dtype,
        )  # [S, T, 16, H, W], float16 on CPU

        # Normalize to row-major shape: add batch axis
        lat_np = lean_ode_t.unsqueeze(0).numpy() if lean_ode_t.ndim in (4, 5) else lean_ode_t.numpy()[None, ...]
        arrays_dict = {
            "latents": lat_np,
            "prompts": np.array([prompts]),
        }
        # Write S rows for 'latents' and 1 row for 'prompts'
        store_arrays_to_lmdb(out_env, arrays_dict, start_index=counter)
        counter += 1

    # Save shapes
    # latents_shape_out = (N, [S,] T, 16, H, W)  — keep same rank as input
    with out_env.begin(write=True) as txn:
        lat_shape = list(latents_shape)
        lat_shape[0] = counter
        # input shape could be (N, T, C, H, W) or (N, S, T, C, H, W)
        if len(lat_shape) == 5:
            # (N, T, C, H, W) -> (N, T_lean, 16, H, W)  (T_lean might differ)
            lat_shape[2] = 16
        elif len(lat_shape) == 6:
            # (N, S, T, C, H, W) -> (N, S, T_lean, 16, H, W)
            lat_shape[3] = 16
        else:
            # fallback: set channel dim to 16 at last-3 position
            lat_shape[-3] = 16
        shape_key = "latents_shape".encode()
        txn.put(shape_key, " ".join(map(str, lat_shape)).encode())

        # prompts shape: (N,)
        prom_shape = (counter,)
        txn.put("prompts_shape".encode(), " ".join(map(str, prom_shape)).encode())

    in_env.close()
    out_env.close()
    print(f"Done. Wrote {counter} samples to {args.output_lmdb}")


if __name__ == "__main__":
    main()


