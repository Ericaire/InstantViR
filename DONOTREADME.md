# InstantViR

## 1. Entry Points and Directory

- Main training entry: `instantvir/train_distillation.py`
- ODE pretraining entry (optional): `instantvir/train_ode.py`
- Minimal inverse-problem inference entry: `minimal_inference/autoregressive_inverse_inference.py`
- Pre-degraded LMDB generation: `instantvir/scripts/create_degraded_dataset.py`
- LMDB shard merge: `instantvir/scripts/merge_lmdb_shards.py`
- Config directory: `configs/`

Common config examples:

- WAN inverse inpainting: `configs/wan_causal_inverse_inpainting.yaml`
- WAN inverse deblur: `configs/wan_causal_inverse_spatial_gaussian.yaml`
- WAN inverse SRx4: `configs/wan_causal_inverse_sr4.yaml`
- LeanVAE inverse inpainting: `configs/wan_causal_inverse_inpainting_leanvae.yaml`
- LeanVAE inverse deblur: `configs/wan_causal_inverse_spatial_gaussian_leanvae.yaml`
- LeanVAE inverse SRx4: `configs/wan_causal_inverse_sr4_leanvae.yaml`

---

## 2. Environment Setup

Run in the repository root:

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR

conda create -n instantvir python=3.10 -y
source /root/miniconda3/bin/activate instantvir

pip install torch torchvision
pip install -r requirements.txt
python setup.py develop
```

Model checkpoint preparation:

- Wan base checkpoint directory: `wan_models/Wan2.1-T2V-1.3B/`
- Training/inference checkpoints (set via `generator_ckpt` in config or `--checkpoint_folder` from CLI)
- If using LeanVAE: `LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt`

---

## 3. Data Formats and Key Concepts

### 3.1 Two LMDB Types

1) **clean latent LMDB** (clean latents + prompts only)
2) **predegraded LMDB** (clean latents + degraded latents + prompts, optional mask)

Inverse-problem training/inference generally uses type 2 (`use_predegraded_dataset: true`).

### 3.2 Task Name Mapping

- inpainting: `inverse_problem_type: inpainting`
- deblur (spatial Gaussian): `inverse_problem_type: spatial_blur`
- SRx4: `inverse_problem_type: super_resolution`

### 3.3 Inference Indexing and Split

`minimal_inference/autoregressive_inverse_inference.py` splits `data_path` into train/val with a default 9:1 ratio (fixed `seed=42`), and `--test_video_index` is the **index in the val split**.

---

## 4. Quick Inference (with Existing predegraded LMDB)

### 4.1 WAN (inpainting / deblur / SRx4)

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

# Inpainting
CUDA_VISIBLE_DEVICES=0 python -m minimal_inference.autoregressive_inverse_inference \
  --config_path configs/wan_causal_inverse_inpainting.yaml \
  --output_folder outputs/infer_inpainting_wan \
  --data_path data/mixkit_latents_inpainting_mask0p5_lmdb \
  --use_predegraded_dataset \
  --checkpoint_folder outputs/wan_causal_inverse_inpainting/<run>/checkpoint_model_<step> \
  --test_video_index 14

# Deblur (spatial gaussian)
CUDA_VISIBLE_DEVICES=0 python -m minimal_inference.autoregressive_inverse_inference \
  --config_path configs/wan_causal_inverse_spatial_gaussian.yaml \
  --output_folder outputs/infer_deblur_wan \
  --data_path data/mixkit_latents_spatial_blur_k61_s3_lmdb \
  --use_predegraded_dataset \
  --checkpoint_folder outputs/wan_causal_inverse_spatial_gaussian/<run>/checkpoint_model_<step> \
  --test_video_index 14

# SRx4
CUDA_VISIBLE_DEVICES=0 python -m minimal_inference.autoregressive_inverse_inference \
  --config_path configs/wan_causal_inverse_sr4.yaml \
  --output_folder outputs/infer_sr4_wan \
  --data_path data/sr4_predegraded_merged.lmdb \
  --use_predegraded_dataset \
  --checkpoint_folder outputs/wan_causal_inverse_sr4/<run>/checkpoint_model_<step> \
  --test_video_index 14
```

### 4.2 LeanVAE (inpainting / deblur / SRx4)

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

# Inpainting
CUDA_VISIBLE_DEVICES=0 python -m minimal_inference.autoregressive_inverse_inference \
  --config_path configs/wan_causal_inverse_inpainting_leanvae.yaml \
  --output_folder outputs/infer_inpainting_leanvae \
  --data_path data/inpainting_leanvae_merged.lmdb \
  --use_predegraded_dataset \
  --checkpoint_folder outputs/wan_causal_inverse_inpainting_leanvae_from_wan_ckpt/<run>/checkpoint_model_<step> \
  --test_video_index 14

# Deblur
CUDA_VISIBLE_DEVICES=0 python -m minimal_inference.autoregressive_inverse_inference \
  --config_path configs/wan_causal_inverse_spatial_gaussian_leanvae.yaml \
  --output_folder outputs/infer_deblur_leanvae \
  --data_path data/spatial_gaussian_leanvae_merged.lmdb \
  --use_predegraded_dataset \
  --checkpoint_folder outputs/wan_causal_inverse_spatial_gaussian_leanvae/<run>/checkpoint_model_<step> \
  --test_video_index 14

# SRx4
CUDA_VISIBLE_DEVICES=0 python -m minimal_inference.autoregressive_inverse_inference \
  --config_path configs/wan_causal_inverse_sr4_leanvae.yaml \
  --output_folder outputs/infer_sr4_leanvae \
  --data_path data/sr4_leanvae_merged.lmdb \
  --use_predegraded_dataset \
  --checkpoint_folder outputs/wan_causal_inverse_sr4_leanvae/<run>/checkpoint_model_<step> \
  --test_video_index 14
```

Inference outputs include:

- `reconstructed_val_XXX.mp4`
- `original_val_XXX.mp4`
- `degraded_val_XXX_upx4.mp4` / `degraded_val_XXX_lr.mp4`

---

## 5. Training Reproduction (InstantViR inverse)

### 5.1 Single-Node Multi-GPU Training (Recommended Entry)

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

torchrun --nproc_per_node=4 -m instantvir.train_distillation \
  --config_path configs/wan_causal_inverse_inpainting.yaml \
  --no_visualize
```

To switch tasks, change only the config (and corresponding data path), for example:

- `configs/wan_causal_inverse_spatial_gaussian.yaml`
- `configs/wan_causal_inverse_sr4.yaml`
- `configs/wan_causal_inverse_inpainting_leanvae.yaml`

### 5.2 Required Fields to Verify in Configs

Open the corresponding `configs/*.yaml` and prioritize checking:

- `data_path`: LMDB path for training
- `output_path`: output directory for logs and checkpoints
- `generator_ckpt`: model initialization checkpoint (can resume from existing ckpt)
- `inverse_problem_type`: task type
- `use_predegraded_dataset`: usually should be `true`
- Task-specific parameters:
  - inpainting: `mask_ratio`
  - deblur: `blur_kernel_size`, `blur_sigma`, `noise_level`
  - SRx4: `downscale_factor`

### 5.3 ODE Pretraining (Optional)

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

torchrun --nproc_per_node=4 -m instantvir.train_ode \
  --config_path configs/wan_causal_ode.yaml \
  --no_save
```

---

## 6. Build predegraded LMDB from Raw Data (Before Training)

`create_degraded_dataset.py` supports:

- Reading clean latents from `--original_lmdb_path`
- Or reading frames directly from `--original_frames_dir`
- Different source/target VAE types (`--source_vae_type` + `--vae_type`), including WAN -> LeanVAE conversion

### 6.1 Single-Task Example (SRx4)

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

CUDA_VISIBLE_DEVICES=0 python instantvir/scripts/create_degraded_dataset.py \
  --config_path configs/wan_causal_inverse_sr4.yaml \
  --original_lmdb_path data/mixkit_latents_lmdb \
  --new_lmdb_path data/sr4_predegraded_shard0.lmdb \
  --degradation_type super_resolution \
  --downscale_factor 4 \
  --source_vae_type wan \
  --vae_type wan
```

### 6.2 LeanVAE Target-Space Example (WAN Source -> LeanVAE)

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

CUDA_VISIBLE_DEVICES=0 python instantvir/scripts/create_degraded_dataset.py \
  --config_path configs/wan_causal_inverse_inpainting_leanvae.yaml \
  --original_lmdb_path data/mixkit_latents_lmdb \
  --new_lmdb_path data/inpainting_leanvae_shard0.lmdb \
  --degradation_type inpainting \
  --mask_ratio 0.5 \
  --source_vae_type wan \
  --vae_type leanvae \
  --leanvae_ckpt_path LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt
```

### 6.3 Merge Multiple Shards

```bash
cd /fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR
source /root/miniconda3/bin/activate instantvir

python instantvir/scripts/merge_lmdb_shards.py \
  --shards_glob "data/inpainting_leanvae_shard*.lmdb" \
  --out_lmdb data/inpainting_leanvae_merged.lmdb
```

---

## 7. Common Troubleshooting

1) `ModuleNotFoundError: No module named 'instantvir'`
Run commands from the repository root, and execute `python setup.py develop` first; if needed, add:
`export PYTHONPATH=/fs-computility-new/UPDZ02_sunhe/suzhexu/InstantViR:$PYTHONPATH`

2) Inference uses the wrong dataset
Check whether CLI `--data_path` overrides `data_path` in the config.
In your current experiment setup, different WAN tasks use different predegraded LMDBs, which is expected.

3) GPU memory not released after interruption
Check for leftover python processes; restart the task only after all related processes have exited.

4) SRx4 resolution mismatch
`autoregressive_inverse_inference.py` upsamples SR input and resets cache based on the actual size of `clean_latent`; you should still keep train/inference resolutions consistent.
