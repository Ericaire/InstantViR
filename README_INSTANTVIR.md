# InstantViR 复现指南（基于本仓库）

本 README 面向当前代码库中 `causvid`、`configs`、`minimal_inference` 三部分，给出一套可直接复现的 **训练 + 推理** 流程（含环境、数据、常见问题）。

> 说明：这里的 “InstantViR” 对应你在本仓库里的 inverse/reconstruction 训练与推理管线（inpainting / deblur / SRx4），核心入口在 `causvid/train_distillation.py` 与 `minimal_inference/autoregressive_inverse_inference.py`。

---

## 1. 代码入口与目录

- 训练主入口：`causvid/train_distillation.py`
- ODE 预训练入口（可选）：`causvid/train_ode.py`
- 逆问题最小推理入口：`minimal_inference/autoregressive_inverse_inference.py`
- 预降质 LMDB 生成：`causvid/scripts/create_degraded_dataset.py`
- LMDB 分片合并：`causvid/scripts/merge_lmdb_shards.py`
- 配置目录：`configs/`

常用配置示例：

- WAN inverse inpainting：`configs/wan_causal_inverse_inpainting.yaml`
- WAN inverse deblur：`configs/wan_causal_inverse_spatial_gaussian.yaml`
- WAN inverse SRx4：`configs/wan_causal_inverse_sr4.yaml`
- LeanVAE inverse inpainting：`configs/wan_causal_inverse_inpainting_leanvae.yaml`
- LeanVAE inverse deblur：`configs/wan_causal_inverse_spatial_gaussian_leanvae.yaml`
- LeanVAE inverse SRx4：`configs/wan_causal_inverse_sr4_leanvae.yaml`

---

## 2. 环境准备

在仓库根目录执行：

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid

conda create -n causvid python=3.10 -y
source /root/miniconda3/bin/activate causvid

pip install torch torchvision
pip install -r requirements.txt
python setup.py develop
```

建议检查：

```bash
python -c "import torch, causvid; print(torch.__version__)"
```

模型权重准备（至少）：

- Wan base 权重目录：`wan_models/Wan2.1-T2V-1.3B/`
- 训练/推理 checkpoint（按 config 的 `generator_ckpt` 或命令行 `--checkpoint_folder`）
- 若使用 LeanVAE：`LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt`

---

## 3. 数据格式与关键概念

### 3.1 两类 LMDB

1) **clean latent LMDB**（仅干净 latent + prompt）  
2) **predegraded LMDB**（干净 latent + 退化 latent + prompt，可选 mask）

逆问题训练/推理一般使用第 2 类（`use_predegraded_dataset: true`）。

### 3.2 任务名对齐

- inpainting：`inverse_problem_type: inpainting`
- deblur（空间高斯）：`inverse_problem_type: spatial_blur`
- SRx4：`inverse_problem_type: super_resolution`

### 3.3 推理索引与划分

`minimal_inference/autoregressive_inverse_inference.py` 默认把 `data_path` 数据按 9:1 切成 train/val（固定 seed=42），`--test_video_index` 是 **val 集索引**。

---

## 4. 快速推理（已有 predegraded LMDB）

以下是最常用复现命令模板。

### 4.1 WAN（inpainting / deblur / SRx4）

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

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

### 4.2 LeanVAE（inpainting / deblur / SRx4）

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

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

推理输出通常包含：

- `reconstructed_val_XXX.mp4`
- `original_val_XXX.mp4`
- `degraded_val_XXX_upx4.mp4` / `degraded_val_XXX_lr.mp4`

---

## 5. 训练复现（InstantViR inverse）

### 5.1 单机多卡训练（推荐入口）

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

torchrun --nproc_per_node=4 -m causvid.train_distillation \
  --config_path configs/wan_causal_inverse_inpainting.yaml \
  --no_visualize
```

换任务只需改 config，例如：

- `configs/wan_causal_inverse_spatial_gaussian.yaml`
- `configs/wan_causal_inverse_sr4.yaml`
- `configs/wan_causal_inverse_inpainting_leanvae.yaml`

### 5.2 配置里必须确认的字段

打开对应 `configs/*.yaml`，优先确认：

- `data_path`：训练 LMDB 路径
- `output_path`：日志和 checkpoint 输出目录
- `generator_ckpt`：初始化模型（可从已有 ckpt 继续）
- `inverse_problem_type`：任务类型
- `use_predegraded_dataset`：通常应为 `true`
- 任务参数：
  - inpainting：`mask_ratio`
  - deblur：`blur_kernel_size`, `blur_sigma`, `noise_level`
  - SRx4：`downscale_factor`

### 5.3 ODE 预训练（可选）

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

torchrun --nproc_per_node=4 -m causvid.train_ode \
  --config_path configs/wan_causal_ode.yaml \
  --no_save
```

---

## 6. 从原始数据构建 predegraded LMDB（训练前）

`create_degraded_dataset.py` 支持：

- 从 `--original_lmdb_path` 读取 clean latent
- 或从 `--original_frames_dir` 直接读帧
- 支持源/目标 VAE 不同（`--source_vae_type` + `--vae_type`），可做 WAN -> LeanVAE 转换

### 6.1 单任务示例（SRx4）

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

CUDA_VISIBLE_DEVICES=0 python causvid/scripts/create_degraded_dataset.py \
  --config_path configs/wan_causal_inverse_sr4.yaml \
  --original_lmdb_path data/mixkit_latents_lmdb \
  --new_lmdb_path data/sr4_predegraded_shard0.lmdb \
  --degradation_type super_resolution \
  --downscale_factor 4 \
  --source_vae_type wan \
  --vae_type wan
```

### 6.2 LeanVAE 目标空间示例（WAN 源 -> LeanVAE）

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

CUDA_VISIBLE_DEVICES=0 python causvid/scripts/create_degraded_dataset.py \
  --config_path configs/wan_causal_inverse_inpainting_leanvae.yaml \
  --original_lmdb_path data/mixkit_latents_lmdb \
  --new_lmdb_path data/inpainting_leanvae_shard0.lmdb \
  --degradation_type inpainting \
  --mask_ratio 0.5 \
  --source_vae_type wan \
  --vae_type leanvae \
  --leanvae_ckpt_path LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt
```

### 6.3 多分片合并

```bash
cd /fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid
source /root/miniconda3/bin/activate causvid

python causvid/scripts/merge_lmdb_shards.py \
  --shards_glob "data/inpainting_leanvae_shard*.lmdb" \
  --out_lmdb data/inpainting_leanvae_merged.lmdb
```

---

## 7. 结果评估（可选）

如果你已导出为帧目录，可用：

```bash
python tools/metrics/compute_psnr_lpips_ssim_dirs.py \
  --recon_dir <recon_frames_dir> \
  --label_dir <gt_frames_dir> \
  --device cuda:0 \
  --save <metrics_txt>
```

FVD（及 SSIM）可用：

```bash
python tools/metrics/compute_ssim_fvd_dirs.py \
  --recon_dir <recon_frames_dir> \
  --label_dir <gt_frames_dir> \
  --device cuda:0 \
  --save <metrics_txt>
```

---

## 8. 常见问题排查

1) `ModuleNotFoundError: No module named 'causvid'`  
请在仓库根目录执行，并先 `python setup.py develop`；必要时补充：  
`export PYTHONPATH=/fs-computility-new/UPDZ02_sunhe/weiminbai/suzhexu/CausVid:$PYTHONPATH`

2) 推理使用了错误数据集  
确认命令行 `--data_path` 是否覆盖了 config 里的 `data_path`。  
你现在这套实验中，WAN 的不同任务来自不同 predegraded LMDB，这是正常的。

3) 进程中断后显存未释放  
检查是否有残留 python 进程；确保所有相关进程退出后再重启任务。

4) SRx4 分辨率不一致  
`autoregressive_inverse_inference.py` 会根据 `clean_latent` 的实际尺寸对 SR 输入做上采样并重置缓存；仍建议保证训练/推理数据分辨率一致。

5) 想看更详细调试信息  
优先前台运行；必要时将 stdout/stderr 重定向到日志文件并实时 `tail -f`。

---

## 9. 一页版最小复现顺序

1. 准备环境（`conda + pip + setup.py develop`）  
2. 准备模型权重（Wan/LeanVAE + 训练 ckpt）  
3. 准备 predegraded LMDB（或直接用已有 LMDB）  
4. `torchrun -m causvid.train_distillation --config_path ...` 训练  
5. `python -m minimal_inference.autoregressive_inverse_inference ...` 推理  
6. 用 `tools/metrics/*.py` 计算指标

如果你只想快速跑通：直接从第 4 节命令开始（已有 LMDB + 已有 checkpoint）。

