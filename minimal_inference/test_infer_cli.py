"""CPU routing regression tests; GPU runtime parity is checked separately."""
import unittest
from minimal_inference.infer import route

class RoutingTests(unittest.TestCase):
    def test_existing_tasks_keep_legacy_arguments(self):
        for task in ('sr4', 'inpainting', 'gaussian-deblur'):
            module, args = route(['--task', task, '--checkpoint_folder', 'ckpt', '--use_predegraded_dataset', '--test_video_index', '14'])
            self.assertEqual(module, 'minimal_inference.autoregressive_inverse_inference')
            self.assertEqual(args[2:], ['--checkpoint_folder', 'ckpt', '--use_predegraded_dataset', '--test_video_index', '14'])
    def test_v2_routes(self):
        module, args = route(['--task', 'sr4', '--input', 'low.mp4', '--checkpoint', 'best.pt', '--output', 'out.mp4'])
        self.assertEqual(module, 'minimal_inference.infer_sr4_gt')
        self.assertEqual(args, ['--input', 'low.mp4', '--checkpoint', 'best.pt', '--output', 'out.mp4'])
    def test_invalid_combinations_fail(self):
        for args in [
            ['--task', 'inpainting', '--input', 'low.mp4'],
            ['--task', 'sr4', '--checkpoint', 'best.pt'],
            ['--task', 'sr4', '--input', 'low.mp4', '--checkpoint', 'best.pt', '--output', 'out.mp4', '--custom_prompt', 'person'],
        ]:
            with self.assertRaises(SystemExit):route(args)

if __name__ == '__main__':unittest.main()
