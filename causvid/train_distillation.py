from causvid.data import ODERegressionLMDBDataset, InverseProblemLMDBDataset, PredegradedLMDBDataset
from causvid.models import get_block_class
from causvid.data import TextDataset
from causvid.util import (
    launch_distributed_job,
    prepare_for_saving,
    set_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_state_dict,
    barrier
)
import torch.distributed as dist
from omegaconf import OmegaConf
from causvid.dmd import DMD
import argparse
import torch
import wandb
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import random_split # Import for splitting dataset


class Trainer:
    def __init__(self, config):
        self.config = config

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        if dist.is_available() and dist.is_initialized():
            global_rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            global_rank = 0
            world_size = 1

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            self.output_path, self.wandb_folder = init_logging_folder(config)
            # 初始化TensorBoard日志
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.output_path, 'tensorboard_logs'))

        # Step 2: Initialize the model and optimizer
        if config.distillation_loss == "dmd":
            self.distillation_model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        if world_size > 1:
            self.distillation_model.generator = fsdp_wrap(
                self.distillation_model.generator,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.generator_fsdp_wrap_strategy,
                transformer_module=(get_block_class(config.generator_fsdp_transformer_module),
                                    ) if config.generator_fsdp_wrap_strategy == "transformer" else None
            )

        if world_size > 1:
            self.distillation_model.real_score = fsdp_wrap(
                self.distillation_model.real_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.real_score_fsdp_wrap_strategy,
                transformer_module=(get_block_class(config.real_score_fsdp_transformer_module),
                                    ) if config.real_score_fsdp_wrap_strategy == "transformer" else None
            )

        if world_size > 1:
            self.distillation_model.fake_score = fsdp_wrap(
                self.distillation_model.fake_score,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.fake_score_fsdp_wrap_strategy,
                transformer_module=(get_block_class(config.fake_score_fsdp_transformer_module),
                                    ) if config.fake_score_fsdp_wrap_strategy == "transformer" else None
            )

        if world_size > 1:
            self.distillation_model.text_encoder = fsdp_wrap(
                self.distillation_model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                transformer_module=(get_block_class(config.text_encoder_fsdp_transformer_module),
                                    ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
            )

        if not config.no_visualize:
            self.distillation_model.vae = self.distillation_model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        # Step 3: Initialize the dataloader

        self.backward_simulation = getattr(config, "backward_simulation", True)

        # Check if we're in inverse problem mode
        self.inverse_problem_mode = getattr(config, "inverse_problem_type", None) is not None
        
        if self.backward_simulation:
            full_dataset = TextDataset(config.data_path)
        else:
            if self.inverse_problem_mode:
                if getattr(config, 'use_predegraded_dataset', False):
                    # Use the new pre-degraded dataset
                    print("--- Using Pre-degraded LMDB Dataset ---")
                    full_dataset = PredegradedLMDBDataset(
                        data_path=config.data_path,
                        max_samples=getattr(config, "max_train_samples", int(1e9))
                    )
                else:
                    # Use the original on-the-fly degradation dataset
                    print("--- Using On-the-fly Degradation Dataset ---")
                    degradation_params = {
                        "blur_kernel_size": getattr(config, "blur_kernel_size", 7),
                        "blur_sigma": getattr(config, 'blur_sigma', 1.5),
                        "noise_level": getattr(config, "noise_level", 0.05),
                        "scale_factor": getattr(config, "scale_factor", 0.25),
                        "mask_ratio": getattr(config, "mask_ratio", 0.5)
                    }
                    full_dataset = InverseProblemLMDBDataset(
                        config.data_path, 
                        inverse_problem_type=config.inverse_problem_type,
                        degradation_params=degradation_params,
                        max_samples=int(1e8)
                    )
            else:
                full_dataset = ODERegressionLMDBDataset(
                    config.data_path, max_pair=int(1e8))

        # Split dataset into training and validation sets (90/10 split)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True, drop_last=True)
            dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.batch_size, sampler=sampler)
        else:
            dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        # If running single process (no dist initialized), save plain state_dict
        if dist.is_available() and dist.is_initialized():
            generator_state_dict = fsdp_state_dict(
                self.distillation_model.generator)
            critic_state_dict = fsdp_state_dict(
                self.distillation_model.fake_score)
        else:
            generator_state_dict = self.distillation_model.generator.state_dict()
            critic_state_dict = self.distillation_model.fake_score.state_dict()
        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self):
        self.distillation_model.eval()  # prevent any randomness (e.g. dropout)

        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        VISUALIZE = self.step % self.config.log_iters == 0 and not self.config.no_visualize

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        degraded_observation = None
        if not self.backward_simulation:
            batch = next(self.dataloader)
            text_prompts = batch["prompts"]
            if self.inverse_problem_mode:
                # For inverse problems, get both clean and degraded latents
                clean_latent = batch["clean_latent"].to(device=self.device, dtype=self.dtype)
                # 兼容双Mask：若存在 fg/bg 两路，则以 tuple 形式传下去用于双重测量一致性
                if ("degraded_observation_fg" in batch) and ("degraded_observation_bg" in batch):
                    degraded_fg = batch["degraded_observation_fg"].to(device=self.device, dtype=self.dtype)
                    degraded_bg = batch["degraded_observation_bg"].to(device=self.device, dtype=self.dtype)
                    # Add batch dimension if missing
                    if clean_latent.ndim == 4:
                        clean_latent = clean_latent.unsqueeze(0)
                    if degraded_fg.ndim == 4:
                        degraded_fg = degraded_fg.unsqueeze(0)
                    if degraded_bg.ndim == 4:
                        degraded_bg = degraded_bg.unsqueeze(0)
                    degraded_observation = (degraded_fg, degraded_bg)
                else:
                    degraded_observation = batch["degraded_observation"].to(device=self.device, dtype=self.dtype)
                    # Add batch dimension if missing
                    if clean_latent.ndim == 4:
                        clean_latent = clean_latent.unsqueeze(0)
                    if degraded_observation.ndim == 4:
                        degraded_observation = degraded_observation.unsqueeze(0)
                # Optional inpainting mask from dataset (pixel space [H,W])
                self.current_inpainting_mask = None
                if "inpainting_mask" in batch:
                    mask = batch["inpainting_mask"].to(device=self.device, dtype=self.dtype)
                    self.distillation_model.current_inpainting_mask = mask
                
                # 注意：若为双Mask，维度处理已在上方完成
            else:
                clean_latent = batch["ode_latent"][:, -1].to(
                    device=self.device, dtype=self.dtype)
        else:
            text_prompts = next(self.dataloader)
            clean_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        
        # Debug output - reduce verbosity
        # if self.step % 100 == 0:  # Only print every 100 steps
        #     print(f"--- Step {self.step}: Data Shapes ---")
        #     print(f"  - text_prompts: {len(text_prompts)} prompts")
        #     if clean_latent is not None:
        #         print(f"  - clean_latent: {clean_latent.shape}")
        #     if degraded_observation is not None:
        #         print(f"  - degraded_observation: {degraded_observation.shape}")
        #     print(f"  - image_or_video_shape: {image_or_video_shape}")


        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.distillation_model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.distillation_model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        if TRAIN_GENERATOR:
            dmd_loss, consistency_loss, generator_log_dict = self.distillation_model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                degraded_observation=degraded_observation
            )

            self.generator_optimizer.zero_grad()
            
            # DMD warmup: optionally mute DMD for the first N steps
            dmd_warmup_steps = getattr(self.config, "dmd_warmup_steps", 0)
            dmd_weight = 0.0 if self.step < dmd_warmup_steps else 1.0
            scaled_dmd_loss = dmd_loss * dmd_weight

            # Single backward: combine losses to avoid retain_graph and donated buffer conflicts
            combined_loss = scaled_dmd_loss + (consistency_loss if consistency_loss is not None else 0.0)
            combined_loss.backward()

            # Clip grads (support both FSDP-wrapped and plain modules)
            if hasattr(self.distillation_model.generator, 'clip_grad_norm_'):
                generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(self.max_grad_norm)
            else:
                from torch.nn.utils import clip_grad_norm_
                generator_grad_norm = clip_grad_norm_(
                    [p for p in self.distillation_model.generator.parameters() if p.requires_grad],
                    self.max_grad_norm
                )
            self.generator_optimizer.step()
            
            # For logging, combine the losses
            generator_loss_val = combined_loss.item()
            
            # Separate logging for individual losses
            dmd_loss_val = scaled_dmd_loss.item()
            consistency_loss_val = consistency_loss.item() if consistency_loss is not None else 0.0
        else:
            generator_log_dict = {}
            generator_loss_val = 0
            dmd_loss_val = 0
            consistency_loss_val = 0

        # Step 4: Train the critic
        critic_loss, critic_log_dict = self.distillation_model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if hasattr(self.distillation_model.fake_score, 'clip_grad_norm_'):
            critic_grad_norm = self.distillation_model.fake_score.clip_grad_norm_(self.max_grad_norm)
        else:
            from torch.nn.utils import clip_grad_norm_
            critic_grad_norm = clip_grad_norm_(
                [p for p in self.distillation_model.fake_score.parameters() if p.requires_grad],
                self.max_grad_norm
            )
        self.critic_optimizer.step()

        # Step 5: Logging
        if self.is_main_process:
            wandb_loss_dict = {
                "critic_loss": critic_loss.item(),
                "critic_grad_norm": critic_grad_norm.item()
            }

            if TRAIN_GENERATOR:
                wandb_loss_dict.update(
                    {
                        "generator_loss": generator_loss_val,
                        "generator_grad_norm": generator_grad_norm.item(),
                        "dmdtrain_gradient_norm": generator_log_dict.get("dmdtrain_gradient_norm", torch.tensor(0.0)).item() if generator_log_dict else 0.0
                    }
                )

            if VISUALIZE:
                self.add_visualization(generator_log_dict, critic_log_dict, wandb_loss_dict)

            # wandb.log(wandb_loss_dict, step=self.step)  # 已禁用wandb
            # === TensorBoard日志 ===
            if TRAIN_GENERATOR:
                self.tensorboard_writer.add_scalar("Loss/generator_loss_total", generator_loss_val, self.step)
                self.tensorboard_writer.add_scalar("Loss/generator_loss_dmd", dmd_loss_val, self.step)
                if consistency_loss is not None:
                    self.tensorboard_writer.add_scalar("Loss/generator_loss_consistency", consistency_loss_val, self.step)
                # 新增：分别记录测量一致性与 gt 一致性，便于拆分观察
                # 兼容旧键和新键：若有未加权或加权值都分别记录
                if generator_log_dict and "measurement_consistency_loss_unweighted" in generator_log_dict:
                    self.tensorboard_writer.add_scalar(
                        "Loss/generator_loss_measurement_consistency_unweighted",
                        generator_log_dict["measurement_consistency_loss_unweighted"],
                        self.step
                    )
                if generator_log_dict and "measurement_consistency_loss_weighted" in generator_log_dict:
                    self.tensorboard_writer.add_scalar(
                        "Loss/generator_loss_measurement_consistency",
                        generator_log_dict["measurement_consistency_loss_weighted"],
                        self.step
                    )
                if generator_log_dict and "gt_consistency_loss" in generator_log_dict:
                    self.tensorboard_writer.add_scalar(
                        "Loss/generator_loss_gt_consistency",
                        generator_log_dict["gt_consistency_loss"],
                        self.step
                    )
                # 记录 chain regularizer（multistep训练的跨步loss）
                if generator_log_dict and "chain_regularizer" in generator_log_dict:
                    self.tensorboard_writer.add_scalar(
                        "Loss/chain_regularizer",
                        generator_log_dict["chain_regularizer"],
                        self.step
                    )
                
                self.tensorboard_writer.add_scalar("Metrics/generator_grad_norm", generator_grad_norm.item(), self.step)
                if generator_log_dict and "dmdtrain_gradient_norm" in generator_log_dict:
                    self.tensorboard_writer.add_scalar("Metrics/dmd_gradient_norm", generator_log_dict["dmdtrain_gradient_norm"].item(), self.step)
            
            self.tensorboard_writer.add_scalar("Loss/critic_loss", critic_loss.item(), self.step)
            self.tensorboard_writer.add_scalar("Metrics/critic_grad_norm", critic_grad_norm.item(), self.step)
            # 记录学习率
            self.tensorboard_writer.add_scalar("Training/learning_rate", self.generator_optimizer.param_groups[0]['lr'], self.step)

    def add_visualization(self, generator_log_dict, critic_log_dict, wandb_loss_dict):
        critictrain_latent, critictrain_noisy_latent, critictrain_pred_image = map(
            lambda x: self.distillation_model.vae.decode_to_pixel(
                x).squeeze(1),
            [critic_log_dict['critictrain_latent'], critic_log_dict['critictrain_noisy_latent'],
                critic_log_dict['critictrain_pred_image']]
        )

        wandb_loss_dict.update({
            "critictrain_latent": prepare_for_saving(critictrain_latent),
            "critictrain_noisy_latent": prepare_for_saving(critictrain_noisy_latent),
            "critictrain_pred_image": prepare_for_saving(critictrain_pred_image)
        })

        if "dmdtrain_clean_latent" in generator_log_dict:
            (dmdtrain_clean_latent, dmdtrain_noisy_latent, dmdtrain_pred_real_image, dmdtrain_pred_fake_image) = map(
                lambda x: self.distillation_model.vae.decode_to_pixel(
                    x).squeeze(1),
                [generator_log_dict['dmdtrain_clean_latent'], generator_log_dict['dmdtrain_noisy_latent'],
                    generator_log_dict['dmdtrain_pred_real_image'], generator_log_dict['dmdtrain_pred_fake_image']]
            )

            wandb_loss_dict.update(
                {
                    "dmdtrain_clean_latent": prepare_for_saving(dmdtrain_clean_latent),
                    "dmdtrain_noisy_latent": prepare_for_saving(dmdtrain_noisy_latent),
                    "dmdtrain_pred_real_image": prepare_for_saving(dmdtrain_pred_real_image),
                    "dmdtrain_pred_fake_image": prepare_for_saving(dmdtrain_pred_fake_image)
                }
            )

    def train(self):
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    # 记录每步训练时间到TensorBoard
                    self.tensorboard_writer.add_scalar("Training/iteration_time", current_time - self.previous_time, self.step)
                    self.previous_time = current_time
            self.step += 1
        # 训练结束时关闭TensorBoard
        if self.is_main_process:
            self.tensorboard_writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = Trainer(config)
    trainer.train()

    # wandb.finish()  # 已禁用wandb


if __name__ == "__main__":
    main()
