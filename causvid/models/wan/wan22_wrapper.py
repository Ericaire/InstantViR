import os
import sys
from typing import List, Optional

import torch

from causvid.models.model_interface import DiffusionModelInterface, TextEncoderInterface
from causvid.models.wan.flow_match import FlowMatchScheduler


# ------------------------ Wan2.2 integration ------------------------ #

# Resolve Wan2.2 repo root (assumed to be cloned as `Wan2.2` under CausVid project root)
_HERE = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
WAN22_ROOT = os.path.join(_PROJECT_ROOT, "Wan2.2")
WAN22_ROOT = os.path.abspath(WAN22_ROOT)

_WAN22_AVAILABLE = os.path.isdir(WAN22_ROOT)
if _WAN22_AVAILABLE and WAN22_ROOT not in sys.path:
    sys.path.append(WAN22_ROOT)

try:
    if _WAN22_AVAILABLE:
        from wan.modules.model import WanModel as Wan22Model  # type: ignore
        from wan.modules.t5 import umt5_xxl as wan22_umt5_xxl  # type: ignore
        from wan.modules.tokenizers import HuggingfaceTokenizer as Wan22Tokenizer  # type: ignore
        from wan.configs.wan_t2v_A14B import t2v_A14B as WAN22_T2V_CONFIG  # type: ignore
    else:
        Wan22Model = None  # type: ignore
        wan22_umt5_xxl = None  # type: ignore
        Wan22Tokenizer = None  # type: ignore
        WAN22_T2V_CONFIG = None  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    Wan22Model = None  # type: ignore
    wan22_umt5_xxl = None  # type: ignore
    Wan22Tokenizer = None  # type: ignore
    WAN22_T2V_CONFIG = None  # type: ignore
    _WAN22_AVAILABLE = False

DEFAULT_WAN22_T2V_CKPT_DIR = (
    os.path.join(WAN22_ROOT, "Wan2.2-T2V-A14B") if _WAN22_AVAILABLE else None
)


class Wan22TextEncoder(TextEncoderInterface):
    """
    文本编码器：使用 Wan2.2 T2V-A14B 的 umt5-xxl 编码器和 tokenizer。

    输出格式与原来的 `WanTextEncoder` 保持一致：
        return {"prompt_embeds": context}
    其中 context 形状为 [B, L, C]。
    """

    def __init__(self, ckpt_dir: Optional[str] = None) -> None:
        super().__init__()

        if not _WAN22_AVAILABLE or wan22_umt5_xxl is None or WAN22_T2V_CONFIG is None:
            raise ImportError(
                "Wan2.2 未正确安装：请确认 CausVid 根目录下存在 `Wan2.2/`，"
                "且其中包含 `wan` 包与 `Wan2.2-T2V-A14B` 权重目录。"
            )

        if ckpt_dir is None:
            if DEFAULT_WAN22_T2V_CKPT_DIR is None:
                raise ValueError(
                    "未找到默认的 Wan2.2-T2V-A14B 权重目录，请通过 "
                    "`--wan2_2_ckpt_dir` 显式指定。"
                )
            ckpt_dir = DEFAULT_WAN22_T2V_CKPT_DIR
        self.ckpt_dir = ckpt_dir

        # 初始化文本编码器
        self.text_encoder = (
            wan22_umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )

        ckpt_path = os.path.join(ckpt_dir, WAN22_T2V_CONFIG.t5_checkpoint)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.text_encoder.load_state_dict(state)

        tokenizer_path = os.path.join(ckpt_dir, WAN22_T2V_CONFIG.t5_tokenizer)
        self.tokenizer = Wan22Tokenizer(
            name=tokenizer_path,
            seq_len=WAN22_T2V_CONFIG.text_len,
            clean="whitespace",
        )

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        # 将 padding 部分置零，保持与 WanTextEncoder 一致
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0

        return {
            "prompt_embeds": context
        }


class Wan22DiffusionWrapper(DiffusionModelInterface):
    """
    Wan2.2 T2V-A14B 的 Diffusion Wrapper。

    - 使用 Wan2.2 的 MoE 结构（high_noise_model / low_noise_model）。
    - 接口保持与 `WanDiffusionWrapper` 一致（输入 / 输出均为 [B, F, C, H, W] 的 x0 预测）。
    - 内部仍使用 flow-matching 形式的 `_convert_flow_pred_to_x0` / `_convert_x0_to_flow_pred`，
      方便与 CausVid 的 ODE/DMD 流程对接。
    """

    def __init__(
        self,
        ckpt_dir: Optional[str] = None,
        shift: Optional[float] = None,
    ):
        super().__init__()

        if not _WAN22_AVAILABLE or Wan22Model is None or WAN22_T2V_CONFIG is None:
            raise ImportError(
                "Wan2.2 未正确安装：请确认 CausVid 根目录下存在 `Wan2.2/`，"
                "且其中包含 `wan` 包与 `Wan2.2-T2V-A14B` 权重目录。"
            )

        if ckpt_dir is None:
            if DEFAULT_WAN22_T2V_CKPT_DIR is None:
                raise ValueError(
                    "未找到默认的 Wan2.2-T2V-A14B 权重目录，请通过 "
                    "`--wan2_2_ckpt_dir` 显式指定。"
                )
            ckpt_dir = DEFAULT_WAN22_T2V_CKPT_DIR
        self.ckpt_dir = ckpt_dir

        # 加载 MoE 的高噪声 / 低噪声 expert
        self.low_noise_model = Wan22Model.from_pretrained(
            ckpt_dir, subfolder=WAN22_T2V_CONFIG.low_noise_checkpoint
        )
        self.high_noise_model = Wan22Model.from_pretrained(
            ckpt_dir, subfolder=WAN22_T2V_CONFIG.high_noise_checkpoint
        )
        self.low_noise_model.eval()
        self.high_noise_model.eval()

        self.uniform_timestep = True

        # FlowMatchScheduler：仅用于 x0 <-> flow_pred 的转换（与 Wan2.1 逻辑保持一致）
        default_shift = float(getattr(WAN22_T2V_CONFIG, "sample_shift", 12.0))
        shift_val = float(shift) if shift is not None else default_shift
        self.scheduler = FlowMatchScheduler(
            shift=shift_val, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # MoE 切换阈值：与官方实现保持一致，使用 boundary * num_train_timesteps
        self.moe_boundary = float(WAN22_T2V_CONFIG.boundary) * float(
            WAN22_T2V_CONFIG.num_train_timesteps
        )

        self.seq_len = None  # 实际上在 forward 中动态计算
        super().post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.low_noise_model.enable_gradient_checkpointing()
        self.high_noise_model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        与 `WanDiffusionWrapper` 保持相同实现。
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        与 `WanDiffusionWrapper._convert_x0_to_flow_pred` 保持相同实现。
        """
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
    ) -> torch.Tensor:
        """
        输入 / 输出均为 [B, F, C, H, W] 的 x0 预测。
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        # 根据当前 timestep 选择 MoE expert
        # 这里假设 batch 内 timestep 一致（对 ODE pairs 生成是成立的）
        t_scalar = input_timestep.detach().float().mean().item()
        if t_scalar >= self.moe_boundary:
            active_model = self.high_noise_model
            inactive_model = self.low_noise_model
        else:
            active_model = self.low_noise_model
            inactive_model = self.high_noise_model

        # --- 简单的 CPU/GPU offload 策略 ---
        # 只保证当前使用的 expert 在输入所在的 GPU 上，
        # 另一侧 expert（inactive_model）如在 GPU 上则挪回 CPU，避免双 expert 同驻导致 OOM。
        device = noisy_image_or_video.device
        if hasattr(inactive_model, "parameters"):
            try:
                if next(inactive_model.parameters()).device.type == "cuda":
                    inactive_model.to("cpu")
            except StopIteration:
                pass

        try:
            if next(active_model.parameters()).device != device:
                active_model.to(device)
        except StopIteration:
            pass

        # 动态计算 seq_len 以匹配 block mask
        B, T, C, H, W = noisy_image_or_video.shape
        patch_size = getattr(active_model, "patch_size", (1, 2, 2))
        if len(patch_size) == 3:
            Hp, Wp = patch_size[1], patch_size[2]
        else:
            Hp, Wp = patch_size[0], patch_size[1]
        frame_seqlen = (H * W) // (Hp * Wp)
        total_length = frame_seqlen * T
        seq_len_dyn = int(total_length)

        # 调用 Wan2.2 backbone
        # Wan2.2 的 WanModel.forward 期望:
        #   - x: List[Tensor]，每个元素 [C_in, F, H, W]
        #   - context: List[Tensor]，每个元素 [L, C_text]
        #   返回: List[Tensor]，每个元素 [C_out, F, H, W]
        x_in = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        x_list = [u for u in x_in]  # len = B
        context_list = [prompt_embeds[i] for i in range(B)]

        if kv_cache is not None:
            try:
                outputs = active_model(
                    x_list,
                    t=input_timestep,
                    context=context_list,
                    seq_len=seq_len_dyn,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    current_end=current_end,
                )
            except TypeError:
                outputs = active_model(
                    x_list,
                    t=input_timestep,
                    context=context_list,
                    seq_len=seq_len_dyn,
                )
        else:
            outputs = active_model(
                x_list,
                t=input_timestep,
                context=context_list,
                seq_len=seq_len_dyn,
            )

        # 将 List[Tensor] -> Tensor [B, C, F, H, W]
        if isinstance(outputs, list):
            flow_pred = torch.stack(outputs, dim=0)
        else:
            flow_pred = outputs
        # [B, C, F, H, W] -> [B, F, C, H, W]
        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)

        # flow_pred -> x0 预测
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return pred_x0


