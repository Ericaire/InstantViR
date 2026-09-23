"""Common CLI for released inverse-task models and supervised SR4 v2."""
import argparse
import runpy
import sys

CONFIGS = {
    'sr4': 'configs/wan_causal_inverse_sr4.yaml',
    'inpainting': 'configs/wan_causal_inverse_inpainting.yaml',
    'gaussian-deblur': 'configs/wan_causal_inverse_spatial_gaussian.yaml',
}


def route(argv):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Video SR4 v2: --task sr4 --input low.mp4 --checkpoint best_ema.pt '
        '--output restored.mp4. Released models: pass the existing LMDB flags.'))
    parser.add_argument('--task', required=True, choices=CONFIGS)
    parser.add_argument('--input', help='LR video; supported by SR4 v2 only')
    parser.add_argument('--checkpoint', help='SR4 v2 best_ema.pt or final_ema.pt')
    parser.add_argument('--output', help='SR4 v2 output video')
    parser.add_argument('--prompt-cache', help='Optional cached empty Wan text embedding')
    args, rest = parser.parse_known_args(argv)
    if any((args.input, args.checkpoint, args.output, args.prompt_cache)):
        if args.task != 'sr4':
            parser.error('Raw video mode currently supports SR4 v2 only; use LMDB flags for this task.')
        if not all((args.input, args.checkpoint, args.output)):
            parser.error('SR4 v2 requires --input, --checkpoint and --output.')
        if rest:
            parser.error('Legacy flags cannot be mixed with SR4 v2: ' + ' '.join(rest))
        forwarded = ['--input', args.input, '--checkpoint', args.checkpoint, '--output', args.output]
        if args.prompt_cache:
            forwarded += ['--prompt-cache', args.prompt_cache]
        return 'minimal_inference.infer_sr4_gt', forwarded
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config_path')
    config_args, _ = config_parser.parse_known_args(rest)
    if config_args.config_path is None:
        rest = ['--config_path', CONFIGS[args.task]] + rest
    else:
        from omegaconf import OmegaConf
        expected = {'sr4': 'super_resolution', 'inpainting': 'inpainting', 'gaussian-deblur': 'spatial_blur'}
        if OmegaConf.load(config_args.config_path).get('inverse_problem_type') != expected[args.task]:
            parser.error('--task and --config_path describe different inverse problems.')
    return 'minimal_inference.autoregressive_inverse_inference', rest


def main(argv=None):
    module, forwarded = route(sys.argv[1:] if argv is None else argv)
    original = sys.argv
    try:
        sys.argv = [module] + forwarded
        runpy.run_module(module, run_name='__main__')
    finally:
        sys.argv = original


if __name__ == '__main__':
    main()
