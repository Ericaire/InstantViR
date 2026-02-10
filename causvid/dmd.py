from causvid.models.model_interface import InferencePipelineInterface
from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper,
    get_inference_pipeline_wrapper
)
from causvid.models.wan.video_operators import add_noise_latent, generate_inpainting_mask
from causvid.loss import get_denoising_loss
import torch.nn.functional as F
from typing import Tuple
from torch import nn
import torch

# 新增：像素域高斯核构造
from causvid.models.wan.video_operators import _get_gaussian_kernel2d


class DMD(nn.Module):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__()

        # Initialize dtype early
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.device = device
        self.args = args

        # Step 1: Initialize all models

        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.real_model_name = getattr(args, "real_name", args.model_name)
        self.fake_model_name = getattr(args, "fake_name", args.model_name)

        self.generator_task_type = getattr(
            args, "generator_task_type", args.generator_task)
        self.real_task_type = getattr(
            args, "real_task_type", args.generator_task)
        self.fake_task_type = getattr(
            args, "fake_task_type", args.generator_task)

        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.generator.set_module_grad(
            module_grad=args.generator_grad
        )
        
        # Convert to appropriate device and dtype
        self.generator = self.generator.to(device=self.device, dtype=self.dtype)

        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu", weights_only=False)[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.real_score = get_diffusion_wrapper(
            model_name=self.real_model_name)()
        self.real_score.set_module_grad(
            module_grad=args.real_score_grad
        )
        self.real_score = self.real_score.to(device=self.device, dtype=self.dtype)

        self.fake_score = get_diffusion_wrapper(
            model_name=self.fake_model_name)()
        self.fake_score.set_module_grad(
            module_grad=args.fake_score_grad
        )
        self.fake_score = self.fake_score.to(device=self.device, dtype=self.dtype)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.text_encoder.requires_grad_(False)
        self.text_encoder = self.text_encoder.to(device=self.device, dtype=self.dtype)

        # Load VAE based on vae_type in config
        vae_type = getattr(args, "vae_type", "wan")
        self.vae_type = vae_type
        disable_latent_conversion = getattr(args, "disable_latent_conversion", False)
        
        if vae_type == "leanvae":
            from causvid.models.leanvae_wrapper import LeanVAEWrapper
            leanvae_ckpt_path = getattr(args, "leanvae_ckpt_path", "LeanVAE-master/LeanVAE-16ch_ckpt/LeanVAE-dim16.ckpt")
            print(f"Loading LeanVAE from {leanvae_ckpt_path}")
            self.vae = LeanVAEWrapper(ckpt_path=leanvae_ckpt_path)
            
            # Optionally load WAN VAE for latent space conversion (teacher使用WAN VAE训练的)
            if not disable_latent_conversion:
                print("Loading WAN VAE for latent space conversion...")
                self.wan_vae = get_vae_wrapper(model_name=args.model_name)()
                self.wan_vae.requires_grad_(False)
                self.wan_vae = self.wan_vae.to(device=self.device, dtype=self.dtype)
                self.use_latent_conversion = True
            else:
                print("⚠️  Latent conversion is DISABLED. DMD loss may not work correctly!")
                self.wan_vae = None
                self.use_latent_conversion = False
        else:
            self.vae = get_vae_wrapper(model_name=args.model_name)()
            self.use_latent_conversion = False
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(device=self.device, dtype=self.dtype)

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: InferencePipelineInterface = None

        # Step 2: Initialize all dmd hyperparameters

        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = args.real_guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        self.scheduler = self.generator.get_scheduler()
        self.denoising_loss_func = get_denoising_loss(
            args.denoising_loss_type)()

        # Add inverse problem configuration
        self.inverse_problem_type = getattr(args, "inverse_problem_type", None)
        self.measurement_consistency_weight = getattr(args, "measurement_consistency_weight", 0.0)
        # Inpainting-specific: compute DMD only on masked (unknown) region
        self.dmd_inpainting_masked_only = getattr(args, "dmd_inpainting_masked_only", False)
        # Optional gt consistency (ground-truth pixel/latent supervision)
        self.use_gt_consistency = getattr(args, "use_gt_consistency", False)
        self.gt_consistency_space = getattr(args, "gt_consistency_space", "latent")  # 'latent' | 'pixel'
        self.gt_consistency_weight = getattr(args, "gt_consistency_weight", 1.0)
        
        if self.inverse_problem_type:
            self.setup_degradation_operator(args)

        # Initialize debug counter
        self._debug_counter = 0

        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda().cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Ensure scheduler has alphas_cumprod available on device for renoise operations
        self._init_scheduler_alphas()

    def _init_scheduler_alphas(self) -> None:
        """Initialize scheduler.alphas_cumprod if missing, placing it on the correct device."""
        try:
            acp = getattr(self.scheduler, "alphas_cumprod", None)
            if acp is None or (torch.is_tensor(acp) and acp.numel() == 0):
                betas = getattr(self.scheduler, "betas", None)
                if betas is None:
                    cfg = getattr(self.scheduler, "config", None)
                    if cfg is not None and hasattr(cfg, "beta_start") and hasattr(cfg, "beta_end") and hasattr(cfg, "num_train_timesteps"):
                        betas = torch.linspace(float(cfg.beta_start), float(cfg.beta_end), int(cfg.num_train_timesteps), device=self.device, dtype=torch.float32)
                    else:
                        # fallback linear schedule using known num_train_timestep
                        betas = torch.linspace(1e-4, 0.02, int(self.num_train_timestep), device=self.device, dtype=torch.float32)
                else:
                    betas = torch.as_tensor(betas, device=self.device, dtype=torch.float32)
                alphas = 1.0 - betas
                acp = torch.cumprod(alphas, dim=0)
            else:
                acp = acp.to(self.device)
            self.scheduler.alphas_cumprod = acp
        except Exception:
            # last-resort fallback
            betas = torch.linspace(1e-4, 0.02, int(self.num_train_timestep), device=self.device, dtype=torch.float32)
            self.scheduler.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def _convert_latent_leanvae_to_wan(self, leanvae_latent: torch.Tensor) -> torch.Tensor:
        """
        Convert LeanVAE latent to WAN VAE latent space.
        This is used to prepare teacher inputs.
        
        Process: LeanVAE latent → decode to pixel → encode to WAN latent
        
        Args:
            leanvae_latent: Tensor of shape [B, F, C, H, W] in LeanVAE latent space
        Returns:
            wan_latent: Tensor of shape [B, F, C, H, W] in WAN VAE latent space
        """
        if not self.use_latent_conversion:
            return leanvae_latent
        
        # Step 1: Decode LeanVAE latent to pixel space
        # Step 2: Encode pixels to WAN VAE latent space
        # Keep dtype consistent throughout - both VAEs are in the same dtype (bfloat16)
        with torch.no_grad():
            pixels = self.vae.decode_video(leanvae_latent)
            # pixels shape: [B, F, C, H, W], range [-1, 1]
            
            wan_latent = self.wan_vae.encode_video(pixels)
            # wan_latent shape: [B, F, C', H', W']
        
        return wan_latent
    
    def _convert_latent_wan_to_leanvae(self, wan_latent: torch.Tensor) -> torch.Tensor:
        """
        Convert teacher prediction from WAN VAE latent space to LeanVAE latent space.
        This is necessary when using LeanVAE as student VAE but teacher was trained with WAN VAE.
        
        Process: WAN latent → decode to pixel → encode to LeanVAE latent
        
        Args:
            wan_latent: Tensor of shape [B, F, C, H, W] in WAN VAE latent space
        Returns:
            leanvae_latent: Tensor of shape [B, F, C, H, W] in LeanVAE latent space
        """
        if not self.use_latent_conversion:
            return wan_latent
        
        # Step 1: Decode WAN latent to pixel space
        # Step 2: Encode pixels to LeanVAE latent space
        # Keep dtype consistent throughout - both VAEs are in the same dtype (bfloat16)
        with torch.no_grad():
            pixels = self.wan_vae.decode_video(wan_latent)
            # pixels shape: [B, F, C, H, W], range [-1, 1]
            
            leanvae_latent = self.vae.encode_video(pixels)
            # leanvae_latent shape: [B, F, C', H', W']
        
        return leanvae_latent

    def _process_timestep(self, timestep: torch.Tensor, type: str) -> torch.Tensor:
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.
            - type: a string indicating the type of the current model (image, bidirectional_video, or causal_video).
        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if type == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif type == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif type == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError("Unsupported model type {}".format(type))

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Debug output only occasionally
        # if hasattr(self, '_debug_counter'):
        #     self._debug_counter += 1
        # else:
        #     self._debug_counter = 0
            
        # if self._debug_counter % 100 == 0:
        #     print(f"    [DMD._compute_kl_grad] noisy_image_or_video shape: {noisy_image_or_video.shape}")
        #     print(f"    [DMD._compute_kl_grad] estimated_clean_image_or_video shape: {estimated_clean_image_or_video.shape}")
        
        # If using latent conversion, we need to convert inputs for teacher to WAN VAE space
        if self.use_latent_conversion:
            # Convert LeanVAE latents to WAN VAE latents for teacher
            noisy_image_or_video_teacher = self._convert_latent_leanvae_to_wan(noisy_image_or_video)
        else:
            noisy_image_or_video_teacher = noisy_image_or_video
        
        # Step 1: Compute the fake score (student operates in LeanVAE space)
        pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # Step 2: Compute the real score (teacher operates in WAN VAE space)
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video_teacher,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video_teacher,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 2.5: Convert teacher prediction from WAN VAE latent to LeanVAE latent if needed
        if self.use_latent_conversion:
            pred_real_image = self._convert_latent_wan_to_leanvae(pred_real_image)

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
            "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
            "dmdtrain_pred_real_image": pred_real_image.detach(),
            "dmdtrain_pred_fake_image": pred_fake_image.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self, image_or_video: torch.Tensor, conditional_dict: dict,
        unconditional_dict: dict, gradient_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            timestep = torch.randint(
                0,
                self.num_train_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )

            timestep = self._process_timestep(
                timestep, type=self.real_task_type)

            # TODO: Add timestep warping
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            # if self._debug_counter % 100 == 0:
            #     print(f"    [DMD.compute_distribution_matching_loss] original_latent shape: {original_latent.shape}")
            #     print(f"    [DMD.compute_distribution_matching_loss] noisy_latent shape: {noisy_latent.shape}")
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = get_inference_pipeline_wrapper(
            self.generator_model_name,
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block
        )

    @torch.no_grad()
    def _consistency_backward_simulation(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        result = self.inference_pipeline.inference_with_trajectory(noise=noise, conditional_dict=conditional_dict)
        return result

    def _run_generator(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.tensor, degraded_observation: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - degraded_observation: for inverse problems, the degraded observation y [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
        """
        # For inverse problems, we want the generator to act as I_φ(y) → x0
        if self.inverse_problem_type and degraded_observation is not None:
            # ==== Utilities for multistep inverse problems ====
            def _renoise_to_t(x0_latent: torch.Tensor, timestep_val: int, eta: float = 0.0, lock_noise: bool = True) -> torch.Tensor:
                """
                Map x0 (latent) to x_t at given timestep using scheduler alphas.
                - If eta==0, reduce to deterministic mapping (ODE-like) with zero noise.
                - If lock_noise=True, reuse one epsilon across frames for temporal coherence.
                """
                at = self.scheduler.alphas_cumprod[int(timestep_val)]
                at = at.to(x0_latent.device)
                if float(eta) == 0.0:
                    eps = torch.zeros_like(x0_latent)
                else:
                    if lock_noise:
                        eps_one = torch.randn(x0_latent.shape[0], 1, x0_latent.shape[2], x0_latent.shape[3], x0_latent.shape[4], device=x0_latent.device, dtype=x0_latent.dtype)
                        eps = eps_one.expand_as(x0_latent)
                    else:
                        eps = torch.randn_like(x0_latent)
                xt = at.sqrt() * x0_latent + ((1.0 - at).clamp(min=0.0).sqrt() * float(eta)) * eps
                return xt

            def _dc_step(x_latent: torch.Tensor, y_latent: torch.Tensor, step_size: float, prox_steps: int = 1) -> torch.Tensor:
                """
                Simple DC via gradient steps on measurement loss in latent space.
                Uses autograd to backprop through forward operator to get A^T(Ax-y).
                """
                x = x_latent
                for _ in range(max(1, int(prox_steps))):
                    x = x.detach().requires_grad_(True)
                    loss_dc = self.compute_measurement_consistency_loss(x, y_latent)
                    grad, = torch.autograd.grad(loss_dc, x, retain_graph=False, create_graph=False, allow_unused=False)
                    if grad is None:
                        break
                    x = (x - float(step_size) * grad).detach()
                return x
            # Inverse problems: support Scheme B (chain self-conditioning) and Scheme A (single-step sampled)
            if getattr(self.args, "inverse_chain_train", False):
                # Scheme B: run a fixed schedule; each intermediate step feeds the next with a detached output
                t_list = getattr(self.args, "inverse_train_timesteps", None)
                if not t_list:
                    # Default: use denoising_step_list excluding 0, in descending order
                    t_list = [int(v) for v in self.denoising_step_list.tolist() if int(v) != 0]
                    t_list = sorted(t_list, reverse=True)
                # schedules and knobs
                lambda_mc_min = float(getattr(self.args, 'lambda_mc_min', 0.1))
                lambda_mc_max = float(getattr(self.args, 'lambda_mc_max', 1.0))
                gamma_dc_min = float(getattr(self.args, 'gamma_dc_min', 0.0))
                gamma_dc_max = float(getattr(self.args, 'gamma_dc_max', 0.25))
                mu_cons_min = float(getattr(self.args, 'mu_cons_min', 0.0))
                mu_cons_max = float(getattr(self.args, 'mu_cons_max', 0.2))
                dc_steps = int(getattr(self.args, 'dc_steps', 1))
                renoise_eta = float(getattr(self.args, 'renoise_eta', 0.0))
                lock_noise = bool(getattr(self.args, 'lock_noise', True))
                use_cross_mono = bool(getattr(self.args, 'use_cross_mono', True))
                use_cross_align_xt = bool(getattr(self.args, 'use_cross_align_xt', False))

                current = degraded_observation
                # For super-resolution, upsample initial input to HR latent grid
                if self.inverse_problem_type == "super_resolution":
                    try:
                        hr_h = int(self.args.image_or_video_shape[-2])
                        hr_w = int(self.args.image_or_video_shape[-1])
                        _, _, _, h_lr, w_lr = current.shape
                        if (h_lr != hr_h) or (w_lr != hr_w):
                            current = F.interpolate(
                                current.flatten(0, 1),
                                size=(hr_h, hr_w),
                                mode='bilinear', align_corners=False
                            ).unflatten(0, current.shape[:2])
                    except Exception:
                        pass

                last_out = None
                num_steps = len(t_list)
                per_step_mc = []
                per_step_cross = []
                prev_residual = None
                prev_x0 = None
                for idx, t_val in enumerate(t_list):
                    prog = idx / max(1, num_steps - 1)
                    lambda_mc = lambda_mc_min + (lambda_mc_max - lambda_mc_min) * prog
                    gamma_dc = gamma_dc_min + (gamma_dc_max - gamma_dc_min) * prog
                    mu_cons = mu_cons_min + (mu_cons_max - mu_cons_min) * prog
                    timestep = torch.full(
                        image_or_video_shape[:2],
                        int(t_val),
                        device=self.device,
                        dtype=torch.long
                    )
                    out_step = self.generator(
                        noisy_image_or_video=current,
                        conditional_dict=conditional_dict,
                        timestep=timestep
                    )
                    # per-step measurement consistency (unweighted scalar)
                    mc_loss_k = self.compute_measurement_consistency_loss(out_step, degraded_observation)

                    # cross-step regularizers
                    cross_loss_k = out_step.new_zeros(())
                    with torch.no_grad():
                        resid_k = self.compute_measurement_consistency_loss(out_step.detach(), degraded_observation.detach())
                    if use_cross_mono and (prev_residual is not None):
                        cross_loss_k = torch.relu(resid_k - prev_residual.detach())
                    if use_cross_align_xt and (prev_x0 is not None):
                        xt_from_prev = _renoise_to_t(prev_x0.detach(), int(t_val), eta=0.0, lock_noise=lock_noise)
                        cross_loss_k = cross_loss_k + F.mse_loss(current.detach(), xt_from_prev)

                    # DC refinement at this step (latent-domain)
                    if gamma_dc > 0.0:
                        out_step = _dc_step(out_step, degraded_observation, gamma_dc, prox_steps=dc_steps)

                    last_out = out_step  # keep the graph for the final step
                    per_step_mc.append(lambda_mc * mc_loss_k)
                    if mu_cons > 0.0:
                        per_step_cross.append(mu_cons * cross_loss_k)
                    prev_residual = resid_k
                    prev_x0 = out_step.detach()
                    if idx < len(t_list) - 1:
                        # self-conditioning input for the next step with time alignment
                        next_t = int(t_list[idx + 1])
                        current = _renoise_to_t(out_step.detach(), next_t, eta=renoise_eta, lock_noise=lock_noise)
                # Use the last step output (with grad) for loss computation
                pred_image_or_video = last_out.type_as(degraded_observation)
                # carry chain loss terms for aggregation in generator_loss
                self._chain_loss_terms = {
                    'per_step_mc': per_step_mc,
                    'per_step_cross': per_step_cross,
                }
            else:
                # Scheme A or default: single-step mapping at a chosen timestep
                if getattr(self.args, "inverse_multi_step_train", False):
                    train_steps = getattr(self.args, "inverse_train_timesteps", None)
                    if train_steps is None or len(train_steps) == 0:
                        # Default to denoising_step_list excluding 0
                        train_steps = [int(v) for v in self.denoising_step_list.tolist() if int(v) != 0]
                    # Uniformly sample one timestep id for this batch
                    timestep_value = int(train_steps[torch.randint(0, len(train_steps), ()).item()])
                else:
                    # Fixed timestep based on problem type (previous default behavior)
                    if self.inverse_problem_type in ("spatial_blur", "temporal_blur", "gaussian_blur", "super_resolution", "gaussian_denoising"):
                        timestep_value = 522
                    else:
                        timestep_value = 522
                        #

                timestep = torch.full(
                    image_or_video_shape[:2],
                    timestep_value,
                    device=self.device,
                    dtype=torch.long
                )

                # For super-resolution, the measurement y is LR latent. Upsample to HR latent grid for the generator.
                gen_input = degraded_observation
                if self.inverse_problem_type == "super_resolution":
                    try:
                        hr_h = int(self.args.image_or_video_shape[-2])
                        hr_w = int(self.args.image_or_video_shape[-1])
                        _, _, _, h_lr, w_lr = gen_input.shape
                        if (h_lr != hr_h) or (w_lr != hr_w):
                            gen_input = F.interpolate(
                                gen_input.flatten(0, 1),
                                size=(hr_h, hr_w),
                                mode='bilinear', align_corners=False
                            ).unflatten(0, gen_input.shape[:2])
                    except Exception:
                        pass

                # 兼容双Mask：若 degraded_observation 为 (fg, bg)，默认以 fg 作为生成器输入
                if isinstance(degraded_observation, (list, tuple)):
                    gen_input_eff = degraded_observation[0]
                else:
                    gen_input_eff = gen_input

                pred_image_or_video = self.generator(
                    noisy_image_or_video=gen_input_eff,
                    conditional_dict=conditional_dict,
                    timestep=timestep
                )

            gradient_mask = None
            # 兼容双Mask：degraded_observation 可能是 (fg, bg) tuple，这里仅取其 dtype 进行对齐
            y_ref = degraded_observation[0] if isinstance(degraded_observation, (list, tuple)) else degraded_observation
            pred_image_or_video = pred_image_or_video.type_as(y_ref)
            return pred_image_or_video, gradient_mask
        
        # Original implementation for standard DMD
        # Step 1: Sample noise and backward simulate the generator's input
        if getattr(self.args, "backward_simulation", True):
            simulated_noisy_input = self._consistency_backward_simulation(
                noise=torch.randn(image_or_video_shape,
                                  device=self.device, dtype=self.dtype),
                conditional_dict=conditional_dict
            )
        else:
            simulated_noisy_input = []
            for timestep in self.denoising_step_list:
                noise = torch.randn(
                    image_or_video_shape, device=self.device, dtype=self.dtype)

                noisy_timestep = timestep * torch.ones(
                    image_or_video_shape[:2], device=self.device, dtype=torch.long)

                if timestep != 0:
                    noisy_image = self.scheduler.add_noise(
                        clean_latent.flatten(0, 1),
                        noise.flatten(0, 1),
                        noisy_timestep.flatten(0, 1)
                    ).unflatten(0, image_or_video_shape[:2])
                else:
                    noisy_image = clean_latent

                simulated_noisy_input.append(noisy_image)

            simulated_noisy_input = torch.stack(simulated_noisy_input, dim=1)

        # Step 2: Randomly sample a timestep and pick the corresponding input
        index = torch.randint(0, len(self.denoising_step_list), [
                              image_or_video_shape[0], image_or_video_shape[1]], device=self.device, dtype=torch.long)
        index = self._process_timestep(index, type=self.generator_task_type)

        # select the corresponding timestep's noisy input from the stacked tensor [B, T, F, C, H, W]
        noisy_input = torch.gather(
            simulated_noisy_input, dim=1,
            index=index.reshape(index.shape[0], 1, index.shape[1], 1, 1, 1).expand(
                -1, -1, -1, *image_or_video_shape[2:])
        ).squeeze(1)

        timestep = self.denoising_step_list[index]

        pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        gradient_mask = None  # timestep != 0

        # pred_image_or_video = noisy_input * \
        #     (1-gradient_mask.float()).reshape(*gradient_mask.shape, 1, 1, 1) + \
        #     pred_image_or_video * gradient_mask.float().reshape(*gradient_mask.shape, 1, 1, 1)

        pred_image_or_video = pred_image_or_video.type_as(noisy_input)

        return pred_image_or_video, gradient_mask

    def generator_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor, degraded_observation: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - degraded_observation: for inverse problems, the degraded observation y [B, F, C, H, W].
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Run generator on backward simulated noisy input
        # 简单的 debug 计数器，便于后面按频率打印信息
        if not hasattr(self, "_debug_counter"):
            self._debug_counter = 0
        self._debug_counter += 1
        # if self._debug_counter % 100 == 0:
        #     print("  [DMD.generator_loss] Entering...")
        #     if clean_latent is not None:
        #         print(f"  [DMD.generator_loss] clean_latent shape: {clean_latent.shape}")
        #     if degraded_observation is not None:
        #         print(f"  [DMD.generator_loss] degraded_observation shape: {degraded_observation.shape}")

        pred_image, gradient_mask = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            degraded_observation=degraded_observation
        )

        # Aggregate chain losses if any (for inverse_chain_train)
        chain_reg = pred_image.new_zeros(())
        if hasattr(self, '_chain_loss_terms') and isinstance(self._chain_loss_terms, dict):
            for term in self._chain_loss_terms.get('per_step_mc', []):
                chain_reg = chain_reg + term
            for term in self._chain_loss_terms.get('per_step_cross', []):
                chain_reg = chain_reg + term
            # clear after consumption to avoid accidental accumulation across steps
            self._chain_loss_terms = {}

        # Optional: for inpainting, restrict DMD to masked region only
        if (
            self.inverse_problem_type == "inpainting"
            and self.dmd_inpainting_masked_only
            and hasattr(self, "current_inpainting_mask")
            and self.current_inpainting_mask is not None
        ):
            try:
                # pred_image: [B,T,C,Hl,Wl]; mask provided in pixel space [B,H,W] or [H,W]
                B, T, C, Hl, Wl = pred_image.shape
                mask = self.current_inpainting_mask
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)  # [1,H,W]
                # Resize to latent spatial size using nearest
                mask_b1hw = F.interpolate(
                    mask.unsqueeze(1).to(device=pred_image.device, dtype=pred_image.dtype),
                    size=(Hl, Wl), mode='nearest'
                ).squeeze(1)  # [B,Hl,Wl]
                # mask_keep==1 means known pixels; we want masked region -> ~keep
                masked_region = mask_b1hw < 0.5
                gradient_mask = masked_region.unsqueeze(1).unsqueeze(2).expand(B, T, C, Hl, Wl)
            except Exception as e:
                # If anything goes wrong, fall back to full-frame DMD
                gradient_mask = None

        # Step 2: Compute the DMD loss)
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask
        )

        # Step 3: Add measurement consistency loss if in inverse problem mode
        consistency_loss = None
        if self.inverse_problem_type and degraded_observation is not None:
            # 兼容双Mask：若 degraded_observation 为 (fg, bg)，则对两路测量分别计算并相加
            if isinstance(degraded_observation, (list, tuple)):
                losses = []
                for y in degraded_observation:
                    losses.append(self.compute_measurement_consistency_loss(pred_image, y))
                meas_loss_unweighted = sum(losses)
            else:
                meas_loss_unweighted = self.compute_measurement_consistency_loss(
                pred_image, degraded_observation
            )
            # Apply weight here to simplify the training loop
            consistency_loss = self.measurement_consistency_weight * meas_loss_unweighted
            dmd_log_dict["measurement_consistency_loss_unweighted"] = meas_loss_unweighted.item()
            dmd_log_dict["measurement_consistency_loss_weighted"] = consistency_loss.item()
            # if self._debug_counter % 100 == 0:
            #     print(f"  [DMD.generator_loss] consistency_loss (weighted): {consistency_loss.item()}")

        # Step 3.1: Optional GT consistency (supervision with clean target)
        gt_consistency_loss = None
        # 记录是否“逻辑上启用了”GT；如果这条始终为 0，说明要么 use_gt_consistency=False，要么 clean_latent=None
        dmd_log_dict["gt_consistency_enabled"] = float(self.use_gt_consistency and (clean_latent is not None))
        if self.use_gt_consistency and (clean_latent is not None):
            # 这里不再 try/except 静默吞掉错误，而是直接抛出，方便你看到真实报错
            if self.gt_consistency_space == "latent":
                # Direct latent-space L2 between prediction and clean latent
                gt_consistency_loss = F.mse_loss(pred_image, clean_latent)
            elif self.gt_consistency_space == "pixel":
                # Decode both to pixel-space, compute pixel MSE
                # 注意：LeanVAE 的权重目前是 bfloat16，因此这里将 latent 转成 self.dtype
                pred_px = self.vae.decode_to_pixel(pred_image.to(self.dtype))
                clean_px = self.vae.decode_to_pixel(clean_latent.to(self.dtype))
                gt_consistency_loss = F.mse_loss(pred_px, clean_px)
            else:
                raise ValueError(f"Unknown gt_consistency_space: {self.gt_consistency_space}")

            # Apply weight
            gt_consistency_loss = gt_consistency_loss * self.gt_consistency_weight
            dmd_log_dict["gt_consistency_loss"] = float(gt_consistency_loss.detach().item())

        # Step 4: Return losses separately for sequential backward pass
        # if self._debug_counter % 100 == 0:
        #     total_loss_val = dmd_loss.item() + (consistency_loss.item() if consistency_loss is not None else 0)
        #     print(f"  [DMD.generator_loss] Returning losses separately. dmd_loss: {dmd_loss.item()}, approx total: {total_loss_val}")
        # Combine measurement consistency (weighted) with optional gt consistency for the second backward
        combined_consistency = None
        if consistency_loss is not None and gt_consistency_loss is not None:
            combined_consistency = consistency_loss + gt_consistency_loss
        elif consistency_loss is not None:
            combined_consistency = consistency_loss
        elif gt_consistency_loss is not None:
            combined_consistency = gt_consistency_loss

        # 合并：主DMD损失
        total_primary = dmd_loss
        
        # 将 chain_reg (包含每一步的 MC loss) 加到 consistency 部分，使其不受 dmd_warmup 影响
        if combined_consistency is None:
            combined_consistency = chain_reg
        else:
            combined_consistency = combined_consistency + chain_reg

        dmd_log_dict["chain_regularizer"] = float(chain_reg.detach().item()) if torch.is_tensor(chain_reg) else 0.0
        return total_primary, combined_consistency, dmd_log_dict

    def critic_loss(self, image_or_video_shape, conditional_dict: dict, unconditional_dict: dict, clean_latent: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        # if self._debug_counter % 100 == 0:
        #     print("  [DMD.critic_loss] Entering...")
        #     if clean_latent is not None:
        #         print(f"  [DMD.critic_loss] clean_latent shape: {clean_latent.shape}")
        with torch.no_grad():
            generated_image, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent
            )


        # Step 2: Compute the fake prediction
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            image_or_video_shape[:2],
            device=self.device,
            dtype=torch.long
        )
        critic_timestep = self._process_timestep(
            critic_timestep, type=self.fake_task_type)

        # TODO: Add timestep warping
        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )
        
        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            assert "wan" in self.args.model_name
            from causvid.models.wan.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 4: TODO: Compute the GAN loss

        # Step 5: Debugging Log
        critic_log_dict = {
            "critictrain_latent": generated_image.detach(),
            "critictrain_noisy_latent": noisy_generated_image.detach(),
            "critictrain_pred_image": pred_fake_image.detach(),
            "critic_timestep": critic_timestep.detach()
        }

        # if self._debug_counter % 100 == 0:
        #     print(f"  [DMD.critic_loss] denoising_loss: {denoising_loss.item()}")
        return denoising_loss, critic_log_dict
    
    def setup_degradation_operator(self, args):
        """Setup forward operator H for inverse problems"""
        if args.inverse_problem_type == "gaussian_blur":
            # For temporal blur
            kernel_size = getattr(args, "blur_kernel_size", 7)
            self.blur_kernel_size = kernel_size
            # We'll apply temporal blur in latent space
            self.forward_operator = self._temporal_blur_latent
        elif args.inverse_problem_type == "spatial_blur":
            # 对齐预降质数据：如使用预降质数据集，则采用像素域模糊→再编码的复合算子
            self.blur_kernel_size = getattr(args, "blur_kernel_size", 7)
            self.blur_sigma = getattr(args, "blur_sigma", 1.5)
            use_pre = getattr(args, "use_predegraded_dataset", False)
            if use_pre:
                self.forward_operator = self._spatial_blur_pixel_roundtrip
            else:
                self.forward_operator = self._spatial_blur_latent
        elif args.inverse_problem_type == "super_resolution":
            self.downscale_factor = getattr(args, "downscale_factor", 4)
            use_pre = getattr(args, "use_predegraded_dataset", False)
            if use_pre:
                # Pixel-space downsample roundtrip; returns LR latent (measurement domain)
                self.forward_operator = self._super_resolution_pixel_roundtrip
            else:
                # Fallback: latent bilinear downsample to LR latent
                self.forward_operator = lambda x: F.interpolate(
                    x.flatten(0, 1),
                    scale_factor=1/self.downscale_factor,
                    mode='bilinear', align_corners=False
                ).unflatten(0, x.shape[:2])
        elif args.inverse_problem_type == "inpainting":
            # If using predegraded dataset, adopt pixel-roundtrip mask using provided per-sample mask
            self.mask_ratio = getattr(args, "mask_ratio", 0.5)
            use_pre = getattr(args, "use_predegraded_dataset", False)
            print(f"[DEBUG] Inpainting setup: use_predegraded_dataset={use_pre}")
            if use_pre:
                # For predegraded dataset, forward operator is identity (degradation already applied)
                # We compare predicted latent directly with pre-degraded latent
                print("[DEBUG] Setting forward_operator to identity for predegraded inpainting")
                self.forward_operator = lambda x: x
            else:
                print("[DEBUG] Setting forward_operator to _random_mask")
                self.forward_operator = self._random_mask
        elif args.inverse_problem_type == "temporal_gaussian":
            # Temporal Gaussian PSF along time; prefer pixel roundtrip to match dataset generation
            self.temporal_kernel_size = getattr(args, "temporal_kernel_size", 7)
            self.temporal_sigma = getattr(args, "temporal_sigma", 1.0)
            use_pre = getattr(args, "use_predegraded_dataset", False)
            if use_pre:
                self.forward_operator = self._temporal_gaussian_pixel_roundtrip
            else:
                # latent fallback
                from causvid.models.wan.video_operators import temporal_gaussian_blur_latent
                self.forward_operator = lambda x: temporal_gaussian_blur_latent(x, kernel_size_t=self.temporal_kernel_size, sigma_t=self.temporal_sigma)
        elif args.inverse_problem_type == "temporal_uniform":
            # Temporal uniform (box) blur along time
            self.temporal_kernel_size = getattr(args, "temporal_kernel_size", 7)
            use_pre = getattr(args, "use_predegraded_dataset", False)
            if use_pre:
                self.forward_operator = self._temporal_uniform_pixel_roundtrip
            else:
                from causvid.models.wan.video_operators import temporal_uniform_blur_latent
                self.forward_operator = lambda x: temporal_uniform_blur_latent(x, kernel_size_t=self.temporal_kernel_size)
        elif args.inverse_problem_type == "gaussian_denoising":
            noise_level = getattr(args, "noise_level", 0.2)
            self.forward_operator = lambda x: add_noise_latent(x, noise_level=noise_level)
        else:
            raise ValueError(f"Unknown inverse problem type: {args.inverse_problem_type}")

    def _apply_inpainting_mask(self, x):
        """Apply inpainting mask."""
        mask = generate_inpainting_mask(x, self.inpainting_mask_type, self.inpainting_box_size)
        return x * mask

    def _spatial_blur_latent(self, x):
        """Apply spatial blur in latent space"""
        from causvid.models.wan.video_operators import spatial_blur_latent
        return spatial_blur_latent(x, kernel_size_s=self.blur_kernel_size, sigma_s=self.blur_sigma)

    def _temporal_blur_latent(self, x):
        """Apply temporal blur in latent space
        x: [B, T, C, H, W] latent tensor
        """
        from causvid.models.wan.video_operators import temporal_blur_latent
        # Ensure we maintain the same shape
        B, T, C, H, W = x.shape
        result = temporal_blur_latent(x, kernel_size_t=self.blur_kernel_size)
        # Ensure result has the same shape as input
        assert result.shape == x.shape, f"Shape mismatch: input {x.shape}, output {result.shape}"
        return result
    
    def _random_mask(self, x):
        """Apply random masking
        x: [B, T, C, H, W] latent tensor
        """
        B, T, C, H, W = x.shape
        # Create random mask
        mask = torch.rand(B, T, 1, H, W, device=x.device) > self.mask_ratio
        return x * mask
    
    def _spatial_blur_pixel_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent → apply spatial Gaussian blur in pixel space → encode back to latent.
        x: [B, T, C, H, W] latent tensor in [-1,1] range
        Returns latent with same shape as x.
        """
        device = x.device
        dtype_model = next(self.vae.model.parameters()).dtype
        
        # Check if using LeanVAE or WAN VAE
        is_leanvae = hasattr(self.vae, 'encode_native')

        # 保存输入以做 STE
        x_in = x

        with torch.no_grad():
            if is_leanvae:
                # LeanVAE: use decode_native/encode_native
                pixel_btchw = self.vae.decode_native(x.to(dtype=dtype_model), use_amp=True)  # [B,T,C,H,W]
                B, T, C, H, W = pixel_btchw.shape
            else:
                # WAN VAE: use model.decode with scale
                mean = self.vae.mean.to(device=device, dtype=dtype_model)
                inv_std = (1.0 / self.vae.std).to(device=device, dtype=dtype_model)
                scale = [mean, inv_std]
                # 1) decode: [B,T,C,H,W] -> [B,C,T,Hs,Ws]
                x_bcthw = x.to(dtype=dtype_model).permute(0,2,1,3,4)
                pixel_bcthw = self.vae.model.decode(x_bcthw, scale).clamp_(-1,1)
                B, C, T, H, W = pixel_bcthw.shape
                pixel_btchw = pixel_bcthw.permute(0,2,1,3,4)  # [B,T,C,H,W]

            # 2) pixel-space Gaussian blur per frame, per channel (keep same dtype as model)
            imgs = pixel_btchw.reshape(B*T, C, H, W)
            kernel = _get_gaussian_kernel2d(int(self.blur_kernel_size), float(self.blur_sigma), dtype=imgs.dtype, device=imgs.device)
            kernel_c = kernel.repeat(C, 1, 1, 1)
            pad = int(self.blur_kernel_size) // 2
            imgs_blur = F.conv2d(imgs, kernel_c, padding=pad, groups=C)
            pixel_blur_btchw = imgs_blur.view(B, T, C, H, W)

            # 3) encode back to latent
            if is_leanvae:
                # LeanVAE: encode_native expects [B,C,T,H,W]
                latent_btchw = self.vae.encode_native(pixel_blur_btchw.permute(0, 2, 1, 3, 4), use_amp=True)  # Input [B,C,T,H,W], output [B,T,16,Hl,Wl]
            else:
                # WAN VAE
                pixel_blur_bcthw = pixel_blur_btchw.permute(0,2,1,3,4)
            latent_bcthw = self.vae.model.encode(pixel_blur_bcthw, scale)
            # to [B,T,C,Hl,Wl]
            latent_btchw = latent_bcthw.permute(0,2,1,3,4)

        # Straight-through estimator: 前向用 latent_btchw，反向让梯度近似恒等传到输入 x
        latent_btchw_ste = x_in + (latent_btchw.to(x_in.dtype) - x_in).detach()
        return latent_btchw_ste

    def _super_resolution_pixel_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent → apply anti-aliased downsample (Resizer, ×1/downscale_factor) in pixel space → encode back.
        返回 LR latent（测量域）；不做 STE，避免与 HR 形状相减。
        x: [B, T, C, H, W] latent in [-1,1]
        """
        device = x.device
        dtype_model = next(self.vae.model.parameters()).dtype
        
        # Check if using LeanVAE or WAN VAE
        is_leanvae = hasattr(self.vae, 'encode_native')

        with torch.no_grad():
            # 1) decode to pixel
            if is_leanvae:
                # LeanVAE: use decode_native, output [B,T,C,H,W]
                pixel_btchw = self.vae.decode_native(x.to(dtype=dtype_model), use_amp=True)
            else:
                # WAN VAE: decode [B,C,T,H,W] and permute to [B,T,C,H,W]
                mean = self.vae.mean.to(device=device, dtype=dtype_model)
                inv_std = (1.0 / self.vae.std).to(device=device, dtype=dtype_model)
                scale = [mean, inv_std]
                x_bcthw = x.to(dtype=dtype_model).permute(0, 2, 1, 3, 4)
                pixel_bcthw = self.vae.model.decode(x_bcthw, scale).clamp_(-1, 1)
                pixel_btchw = pixel_bcthw.permute(0, 2, 1, 3, 4)

            # 2) center-crop to be divisible by factor (safety)
            B, T, C, H, W = pixel_btchw.shape
            f = int(getattr(self, 'downscale_factor', 4))
            Hn = (H // f) * f
            Wn = (W // f) * f
            if Hn != H or Wn != W:
                h0 = (H - Hn) // 2
                w0 = (W - Wn) // 2
                pixel_btchw = pixel_btchw[:, :, :, h0:h0+Hn, w0:w0+Wn]

            # 3) downsample with Resizer if available, else bilinear antialias
            try:
                import sys, os
                util_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "diffusion-posterior-sampling"))
                if util_path not in sys.path:
                    sys.path.append(util_path)
                from util.resizer import Resizer  # type: ignore
                resizer = Resizer(list(pixel_btchw.shape), 1.0 / f)
                # move buffers to correct device
                resizer = resizer.to(device)
                pixel_down = resizer(pixel_btchw)
            except Exception:
                pixel_down = F.interpolate(
                    pixel_btchw.reshape(B*T, C, Hn, Wn),
                    scale_factor=1.0 / f, mode='bilinear', align_corners=False, antialias=True
                ).view(B, T, C, Hn//f, Wn//f)

            # 4) encode back to latent
            if is_leanvae:
                # LeanVAE: encode_native expects [B,C,T,H,W], output [B,T,16,Hl,Wl]
                latent_btchw = self.vae.encode_native(pixel_down.permute(0, 2, 1, 3, 4), use_amp=True)
            else:
                # WAN VAE: [B,T,C,Hd,Wd] -> [B,C,T,Hd,Wd] -> latent [B,T,C,Hl,Wl]
                enc_bcthw = self.vae.model.encode(pixel_down.permute(0, 2, 1, 3, 4), scale)
                latent_btchw = enc_bcthw.permute(0, 2, 1, 3, 4)

        return latent_btchw.to(x.dtype)

    def _inpainting_pixel_roundtrip(self, x: torch.Tensor, mask_hw: torch.Tensor) -> torch.Tensor:
        """
        Decode latent -> apply binary mask in pixel space (0 drop) -> encode back to latent (STE for grads).
        x: [B, T, C, H, W] latent; mask_hw: [H, W] with {0,1} meaning keep mask.
        """
        device = x.device
        dtype_model = next(self.vae.model.parameters()).dtype
        
        # Check if using LeanVAE or WAN VAE
        is_leanvae = hasattr(self.vae, 'encode_native')

        x_in = x
        with torch.no_grad():
            if is_leanvae:
                # LeanVAE: use decode_native/encode_native
                pixel_btchw = self.vae.decode_native(x.to(dtype=dtype_model), use_amp=True)  # [B,T,C,H,W]
                B, T, C, H, W = pixel_btchw.shape
            else:
                # WAN VAE: use model.decode with scale
                mean = self.vae.mean.to(device=device, dtype=dtype_model)
                inv_std = (1.0 / self.vae.std).to(device=device, dtype=dtype_model)
                scale = [mean, inv_std]
                x_bcthw = x.to(dtype=dtype_model).permute(0, 2, 1, 3, 4)
                pixel_bcthw = self.vae.model.decode(x_bcthw, scale).clamp_(-1, 1)
                B, C, T, H, W = pixel_bcthw.shape
                pixel_btchw = pixel_bcthw.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
            
            # Broadcast mask to [B,T,C,H,W]
            mask_t = mask_hw.to(device=pixel_btchw.device, dtype=pixel_btchw.dtype)
            if mask_t.dim() == 2:
                m = mask_t.view(1, 1, 1, H, W).expand(B, T, C, H, W)
            elif mask_t.dim() == 3:
                Bm, Hm, Wm = mask_t.shape
                assert Hm == H and Wm == W, f"Mask spatial mismatch: mask {(Hm,Wm)} vs pixel {(H,W)}"
                assert Bm == B or Bm == 1, f"Mask batch {Bm} incompatible with B={B}"
                m = mask_t.view(Bm, 1, 1, H, W).expand(B, T, C, H, W)
            else:
                raise ValueError(f"Invalid inpainting mask dims: {mask_t.shape}")
            
            pixel_masked = pixel_btchw * m
            
            if is_leanvae:
                # LeanVAE: encode_native expects [B,T,C,H,W]
                latent_btchw = self.vae.encode_native(pixel_masked.permute(0, 2, 1, 3, 4), use_amp=True)  # Input [B,C,T,H,W], output [B,T,16,Hl,Wl]
            else:
                # WAN VAE
                pixel_masked_bcthw = pixel_masked.permute(0, 2, 1, 3, 4)
                latent_bcthw = self.vae.model.encode(pixel_masked_bcthw, scale)
            latent_btchw = latent_bcthw.permute(0, 2, 1, 3, 4)
        
        return x_in + (latent_btchw.to(x_in.dtype) - x_in).detach()

    def _temporal_gaussian_pixel_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent -> apply temporal Gaussian blur in pixel space -> encode back (STE) to latent.
        x: [B, T, C, H, W]
        """
        device = x.device
        dtype_model = next(self.vae.model.parameters()).dtype
        
        # Check if using LeanVAE or WAN VAE
        is_leanvae = hasattr(self.vae, 'encode_native')

        x_in = x
        with torch.no_grad():
            if is_leanvae:
                # LeanVAE: use wrapper methods
                pixel_btchw = self.vae.decode_native(x.to(dtype=dtype_model), use_amp=True)  # [B,T,C,H,W]
                B, T, C, H, W = pixel_btchw.shape
            else:
                # WAN VAE: use model.decode with scale
                mean = self.vae.mean.to(device=device, dtype=dtype_model)
                inv_std = (1.0 / self.vae.std).to(device=device, dtype=dtype_model)
                scale = [mean, inv_std]
                # [B,C,T,H,W]
                x_bcthw = x.to(dtype=dtype_model).permute(0, 2, 1, 3, 4)
                pixel_bcthw = self.vae.model.decode(x_bcthw, scale).clamp_(-1, 1)
                # apply temporal Gaussian blur along T in pixel space
                B, C, T, H, W = pixel_bcthw.shape
                pixel_btchw = pixel_bcthw.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
            
            # Apply temporal Gaussian blur
            seq = pixel_btchw  # [B,T,C,H,W]
            # reuse dataset-side implementation shape-wise
            # reshape to [B*C*H*W, 1, T]
            xc = seq.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, 1, T)
            ksz = int(getattr(self, 'temporal_kernel_size', 7))
            sig = float(getattr(self, 'temporal_sigma', 1.0))
            k = torch.arange(ksz, dtype=pixel_btchw.dtype, device=pixel_btchw.device) - (ksz - 1) / 2
            g = torch.exp(-0.5 * (k / sig) ** 2)
            g = g / g.sum().clamp(min=1e-8)
            kernel = g.view(1, 1, ksz)
            pad = ksz // 2
            xc_pad = F.pad(xc, (pad, pad), mode='replicate')
            xc_blur = F.conv1d(xc_pad, kernel, groups=1)
            pixel_blur_btchw = xc_blur.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)
            
            # encode back
            if is_leanvae:
                # LeanVAE: encode_native expects [B,C,T,H,W], output [B,T,16,Hl,Wl]
                latent_btchw = self.vae.encode_native(pixel_blur_btchw.permute(0, 2, 1, 3, 4), use_amp=True)
            else:
                # WAN VAE
                enc_bcthw = self.vae.model.encode(pixel_blur_btchw.permute(0, 2, 1, 3, 4), scale)
                latent_btchw = enc_bcthw.permute(0, 2, 1, 3, 4)
        return x_in + (latent_btchw.to(x_in.dtype) - x_in).detach()

    def _temporal_uniform_pixel_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent -> apply temporal uniform (box) blur in pixel space -> encode back (STE) to latent.
        x: [B, T, C, H, W]
        """
        device = x.device
        dtype_model = next(self.vae.model.parameters()).dtype
        
        # Check if using LeanVAE or WAN VAE
        is_leanvae = hasattr(self.vae, 'encode_native')

        x_in = x
        with torch.no_grad():
            if is_leanvae:
                # LeanVAE: use wrapper methods
                pixel_btchw = self.vae.decode_native(x.to(dtype=dtype_model), use_amp=True)  # [B,T,C,H,W]
                B, T, C, H, W = pixel_btchw.shape
            else:
                # WAN VAE: use model.decode with scale
                mean = self.vae.mean.to(device=device, dtype=dtype_model)
                inv_std = (1.0 / self.vae.std).to(device=device, dtype=dtype_model)
                scale = [mean, inv_std]
                # [B,C,T,H,W]
                x_bcthw = x.to(dtype=dtype_model).permute(0, 2, 1, 3, 4)
                pixel_bcthw = self.vae.model.decode(x_bcthw, scale).clamp_(-1, 1)
                # apply temporal uniform blur along T in pixel space
                B, C, T, H, W = pixel_bcthw.shape
                pixel_btchw = pixel_bcthw.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
            
            # Apply temporal uniform blur
            seq = pixel_btchw  # [B,T,C,H,W]
            xc = seq.permute(0, 2, 3, 4, 1).contiguous().view(B * C * H * W, 1, T)
            ksz = int(getattr(self, 'temporal_kernel_size', 7))
            kernel = torch.ones(1, 1, ksz, dtype=pixel_btchw.dtype, device=pixel_btchw.device) / float(ksz)
            pad = ksz // 2
            xc_pad = F.pad(xc, (pad, pad), mode='replicate')
            xc_blur = F.conv1d(xc_pad, kernel, groups=1)
            pixel_blur_btchw = xc_blur.view(B, C, H, W, T).permute(0, 4, 1, 2, 3)
            
            # encode back
            if is_leanvae:
                # LeanVAE: encode_native expects [B,C,T,H,W], output [B,T,16,Hl,Wl]
                latent_btchw = self.vae.encode_native(pixel_blur_btchw.permute(0, 2, 1, 3, 4), use_amp=True)
            else:
                # WAN VAE
                enc_bcthw = self.vae.model.encode(pixel_blur_btchw.permute(0, 2, 1, 3, 4), scale)
                latent_btchw = enc_bcthw.permute(0, 2, 1, 3, 4)
        return x_in + (latent_btchw.to(x_in.dtype) - x_in).detach()

    def compute_measurement_consistency_loss(self, pred_image, degraded_observation):
        """Compute ||H(pred_image) - y||^2"""
        # Apply forward operator to prediction
        pred_measurement = self.forward_operator(pred_image)
        
        # Ensure both tensors have the same shape
        if pred_measurement.shape != degraded_observation.shape:
            # Handle potential shape mismatches in super-resolution
            if self.inverse_problem_type == "super_resolution":
                # Upsample degraded_observation back to original size for comparison
                B, T, C, H, W = pred_image.shape
                degraded_observation_upsampled = F.interpolate(
                    degraded_observation.flatten(0, 1), 
                    size=(H, W), 
                    mode='bilinear',
                    align_corners=False
                ).unflatten(0, (B, T))
                
                # Compare in original resolution space
                consistency_loss = F.mse_loss(pred_image, degraded_observation_upsampled)
            else:
                raise ValueError(f"Shape mismatch in consistency loss: pred_measurement {pred_measurement.shape} vs degraded_observation {degraded_observation.shape}")
        else:
            # Compute consistency loss in measurement space
            consistency_loss = F.mse_loss(pred_measurement, degraded_observation)
        
        return consistency_loss
