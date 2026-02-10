from causvid.models.wan.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from causvid.models.wan.flow_match import FlowMatchScheduler
from causvid.util import launch_distributed_job
from causvid.data import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(device, args):
    """
    根据参数选择 Wan2.1 或 Wan2.2 作为 teacher，并返回:
        - model: Diffusion wrapper
        - encoder: 文本编码器
        - scheduler: FlowMatchScheduler（驱动 ODE 轨迹的时间步）
        - unconditional_dict: 用于 CFG 的负向条件
    """
    teacher = getattr(args, "teacher", "wan2.1").lower()
    if teacher not in ["wan2.1", "wan2.2"]:
        raise ValueError(f"Unsupported teacher: {teacher}")

    if teacher == "wan2.1":
        # Wan2.1 1.3B 体量较小，直接使用 float32 方便数值稳定
        model = WanDiffusionWrapper().to(device).to(torch.float32)
        encoder = WanTextEncoder().to(device).to(torch.float32)
        default_shift = 8.0
    else:
        # Wan2.2 A14B MoE 体量巨大：
        # - 不在 wrapper 上调用 .to(device)，避免一次性把两个 expert 都搬上 GPU；
        # - 文本编码器保持在 CPU 上运行（类似官方 t5_cpu=True 的行为）。
        from causvid.models.wan.wan22_wrapper import Wan22DiffusionWrapper, Wan22TextEncoder

        model = Wan22DiffusionWrapper(
            ckpt_dir=args.wan2_2_ckpt_dir,
            shift=args.shift,
        )
        encoder = Wan22TextEncoder(
            ckpt_dir=args.wan2_2_ckpt_dir,
        )
        default_shift = 12.0

    # 冻结 teacher 参数
    if teacher == "wan2.1":
        model.set_module_grad({"model": False})
    else:
        # Wan2.2: 冻结两个 expert
        model.set_module_grad(
            {
                "low_noise_model": False,
                "high_noise_model": False,
            }
        )

    # FlowMatchScheduler：用于生成 ODE 轨迹（noise -> clean）
    shift_val = args.shift if args.shift is not None else default_shift
    scheduler = FlowMatchScheduler(
        shift=shift_val, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(text_prompts=[sample_neg_prompt])
    # 确保条件张量与模型在同一 device 上
    for k, v in list(unconditional_dict.items()):
        if isinstance(v, torch.Tensor):
            unconditional_dict[k] = v.to(device)

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--teacher",
        type=str,
        default="wan2.1",
        choices=["wan2.1", "wan2.2"],
        help="选择 ODE teacher 模型版本：wan2.1 或 wan2.2。",
    )
    parser.add_argument(
        "--wan2_2_ckpt_dir",
        type=str,
        default=None,
        help=(
            "Wan2.2-T2V-A14B checkpoint 目录（当 teacher=wan2.2 时使用）。"
            "若不指定，则默认为 CausVid 根目录下的 Wan2.2/Wan2.2-T2V-A14B。"
        ),
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="FlowMatchScheduler 的 shift 参数；默认 8.0 (wan2.1) / 12.0 (wan2.2)。",
    )
    parser.add_argument(
        "--index_offset",
        type=int,
        default=0,
        help="用于多进程/多分片运行时的样本索引偏移量，避免不同进程写入相同 pt 文件名。",
    )

    args = parser.parse_args()

    # 初始化分布式（WORLD_SIZE>1 时），否则单进程模式
    launch_distributed_job()
    if dist.is_available() and dist.is_initialized():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        global_rank = 0
        world_size = 1

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(
        device=device, args=args)

    dataset = TextDataset(args.caption_path)

    if global_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    for index in tqdm(
        range(int(math.ceil(len(dataset) / world_size))),
        disable=(global_rank != 0),
    ):
        # 先计算当前进程负责的样本在本 shard 中的索引
        dataset_index = index * world_size + global_rank
        if dataset_index >= len(dataset):
            continue

        prompt = dataset[dataset_index]
        # 再加上 index_offset 仅用于 pt 文件命名，避免不同 shard/进程之间重名
        prompt_index = dataset_index + args.index_offset

        # 严格 resume：若目标 pt 已存在，则跳过该样本，避免重复计算
        out_path = os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        if os.path.exists(out_path):
            continue

        conditional_dict = encoder(text_prompts=prompt)
        for k, v in list(conditional_dict.items()):
            if isinstance(v, torch.Tensor):
                conditional_dict[k] = v.to(device)

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, 21], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        noisy_inputs = noisy_inputs[:, [0, 36, 44, -1]]

        stored_data = noisy_inputs

        torch.save(
            {prompt: stored_data.cpu().detach()},
            out_path,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
