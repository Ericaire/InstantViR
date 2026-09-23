"""Inference for v2 checkpoints: pixel Bicubic upsample BEFORE Wan encoding."""
import argparse
import json
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from minimal_inference.sr4_v2 import FORMAT,make_generator,streamed,decode,encode,empty_context
from instantvir.models.wan.wan_wrapper import WanVAEWrapper

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='A real low-resolution MP4, not the HR reference')
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--prompt-cache',default=None,help='Optional cached empty Wan embedding; generated from T5 when omitted')
    a=p.parse_args();torch.set_num_threads(4)
    reader=imageio.get_reader(a.input);fps=float(reader.get_meta_data()['fps']);frames=np.stack([f for f in reader]);reader.close()
    n,h,w,_=frames.shape
    if Path(a.input).resolve()==Path(a.output).resolve():raise ValueError('Input and output paths must differ')
    if n == 0 or h%4 or w%4:raise ValueError('Nonempty video with LR width and height multiples of 4 required')
    pixels=torch.from_numpy(frames.copy()).permute(0,3,1,2).cuda().float()/127.5-1
    padded=1+4*((n-1+3)//4)
    if padded>n:pixels=torch.cat([pixels,pixels[-1:].expand(padded-n,-1,-1,-1)])
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if state.get('format') != FORMAT:raise ValueError('Expected SR4 v2 checkpoint; released checkpoints use the legacy LMDB mode')
    g=make_generator();g.load_state_dict(state['generator'],strict=True);del state
    vae=WanVAEWrapper().cuda().float().eval()
    context=empty_context(a.prompt_cache)
    torch.cuda.synchronize();start=time.time()
    up=F.interpolate(pixels,size=(h*4,w*4),mode='bicubic',align_corners=False).clamp(-1,1)
    condition=encode(vae,up[None])
    prediction=decode(vae,streamed(g,condition,context))[0,:n]
    torch.cuda.synchronize();seconds=time.time()-start
    result=((prediction.clamp(-1,1)+1)*127.5).round().byte().permute(0,2,3,1).cpu().numpy()
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(str(dest),fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
        for frame in result:writer.append_data(frame)
    info={'format':FORMAT,'input':a.input,'checkpoint':a.checkpoint,'frames':n,'padded_frames':padded,'input_size':[w,h],'output_size':[4*w,4*h],'fps':fps,'encode_generator_decode_seconds':seconds}
    dest.with_suffix('.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)

if __name__=='__main__':main()
