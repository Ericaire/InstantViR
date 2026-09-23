"""Inference-only runtime for the supervised SR4 v2 checkpoint contract.

Keep the inverse-std encoding local: legacy tasks retain their original behavior.
"""
import torch
from instantvir.models.wan.wan_wrapper import CausalWanDiffusionWrapper, WanVAEWrapper, WanTextEncoder

FORMAT = 'sr4_gt_v2_hr_bicubic_encoded'


def make_generator():
    g = CausalWanDiffusionWrapper()
    g.model.num_frame_per_block = 3
    return g.cuda().float().eval().requires_grad_(False)


@torch.no_grad()
def encode(vae, pixels):
    mean = vae.mean.to(device=pixels.device, dtype=pixels.dtype)
    inv_std = (1.0 / vae.std).to(device=pixels.device, dtype=pixels.dtype)
    with torch.autocast('cuda', dtype=torch.float16):
        z = vae.model.encode(pixels.permute(0, 2, 1, 3, 4).contiguous(), [mean, inv_std])
    return z.permute(0, 2, 1, 3, 4).half().float()


@torch.no_grad()
def empty_context(cache=None):
    if cache:
        context = torch.load(cache, map_location='cpu', weights_only=True)
        if not isinstance(context, torch.Tensor) or tuple(context.shape) != (1, 512, 4096):
            raise ValueError('Expected an empty Wan text embedding with shape [1,512,4096].')
        if not torch.isfinite(context).all():
            raise ValueError('Non-finite prompt embedding.')
        return context.cuda().float()
    encoder = WanTextEncoder().cuda().eval().requires_grad_(False)
    context = encoder(text_prompts=[''])['prompt_embeds'].half().float()
    del encoder
    torch.cuda.empty_cache()
    return context


@torch.no_grad()
def streamed(g,condition,context):
    b,t,c,h,w=condition.shape
    m=g.model;heads=m.num_heads;d=m.dim//heads;seq=h*w//4
    caches=[{'k':torch.zeros(b,t*seq,heads,d,device='cuda',dtype=torch.bfloat16),
             'v':torch.zeros(b,t*seq,heads,d,device='cuda',dtype=torch.bfloat16)} for _ in m.blocks]
    cross=[{'k':torch.zeros(b,m.text_len,heads,d,device='cuda',dtype=torch.bfloat16),
            'v':torch.zeros(b,m.text_len,heads,d,device='cuda',dtype=torch.bfloat16),'is_init':False} for _ in m.blocks]
    result=[]
    for start in range(0,t,3):
        end=min(start+3,t)
        kw=dict(noisy_image_or_video=condition[:,start:end],conditional_dict={'prompt_embeds':context},
                timestep=torch.full((b,end-start),522,device='cuda',dtype=torch.long),
                kv_cache=caches,crossattn_cache=cross,current_start=start*seq,current_end=end*seq)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=g(**kw)
        result.append(out)
    return torch.cat(result,dim=1)

def decode(vae,z):
    # VAE parameters stay frozen, but decoding predictions must keep input grads.
    with torch.autocast('cuda',dtype=torch.float16):
        return vae.decode_to_pixel(z.float()).float()

