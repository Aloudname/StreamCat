#!/usr/bin/env python3
"""
run.py — Entry point for the real-time streaming inference pipeline.

Usage examples:
    # Default config from config/config.yaml
    python run.py

    # Custom config by arg cmd.
    python run.py --custom/-c

    # Override: use MONAI backend
    python run.py -c --backend/-b monai --model-runtime/-mr onnx

    # Override: use video file as input
    python run.py --source/-s /path/to/video.mp4

    # Override: display width and stride
    python run.py --stride/-stride 8 --max-width/-mw 1920
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from pipeline import StreamPipeline, tprint


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time streaming inference pipeline for LoLA_hsViT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument('--custom', '-c', action='store_true',
                        help='use custom settings')

    # Capture overrides
    parser.add_argument('--source', '-s', type=str, default=None,
                        help='Capture source: device index (0,1,..) or video path')
    parser.add_argument('--env-profile', '-ep', type=str, default=None,
                        choices=['server_test', 'edge'],
                        help='Runtime environment profile')
    parser.add_argument('--input-mode', '-im', type=str, default=None,
                        choices=['opencv', 'npy_stream', 'hsi_camera'],
                        help='Capture input mode')
    parser.add_argument('--npy-dir', type=str, default=None,
                        help='Directory containing .npy files for stream testing')
    parser.add_argument('--npy-glob', type=str, default=None,
                        help='Glob pattern for npy files (default: *.npy)')
    parser.add_argument('--npy-fps', type=float, default=None,
                        help='Playback fps for npy stream mode')
    parser.add_argument('--npy-loop', action='store_true',
                        help='Loop npy files when reaching the end')

    # Inference overrides
    parser.add_argument('--backend', '-b', type=str, default=None,
                        choices=['monai', 'onnx', 'grpc'],
                        help='Inference backend (default: from config)')
    parser.add_argument('--model-path', '-p', type=str, default=None,
                        help='Path to model file (.onnx or .ts)')
    parser.add_argument('--model-runtime', '-mr', type=str, default=None,
                        choices=['onnx', 'torchscript'],
                        help='Runtime used by MONAI backend')
    parser.add_argument('--device', '-d', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Inference device for MONAI backend')
    parser.add_argument('--model-channels', '-mc', type=int, default=None,
                        help='Expected input channels for model (for channel adaptation)')
    parser.add_argument('--grpc-target', '-gt', type=str, default=None,
                        help='gRPC server target (used with --backend grpc)')
    parser.add_argument('--batch-size', '-bs', type=int, default=None,
                        help='Inference batch size')

    # Preprocess overrides
    parser.add_argument('--stride', '-st', type=int, default=None,
                        help='Sliding window stride (smaller = denser)')

    # Display overrides
    parser.add_argument('--max-width', '-mw', type=int, default=None,
                        help='Max display width in pixels')
    parser.add_argument('--alpha', '-a', type=float, default=None,
                        help='Overlay transparency (0.0 - 1.0)')
    parser.add_argument('--headless', action='store_true',
                        help='Disable OpenCV GUI display (for server/no-display env)')

    return parser.parse_args()


def apply_overrides(cfg, args):
    """Override config values from CLI arguments."""
    if not hasattr(cfg, 'runtime') or cfg.runtime is None:
        cfg.runtime = {}

    if args.env_profile is not None:
        cfg.runtime.env_profile = args.env_profile

    if args.input_mode is not None:
        cfg.capture.input_mode = args.input_mode

    if args.npy_dir is not None:
        cfg.capture.npy_dir = args.npy_dir
    if args.npy_glob is not None:
        cfg.capture.npy_glob = args.npy_glob
    if args.npy_fps is not None:
        cfg.capture.npy_fps = args.npy_fps
    if args.npy_loop:
        cfg.capture.npy_loop = True

    # Apply profile defaults unless explicitly overridden
    if cfg.runtime.get('env_profile', 'server_test') == 'server_test' and args.input_mode is None:
        cfg.capture.input_mode = 'npy_stream'
    if cfg.runtime.get('env_profile', 'server_test') == 'edge' and args.input_mode is None:
        cfg.capture.input_mode = 'hsi_camera'

    if args.source:
        # Try to parse as int (device index), otherwise treat as file path
        try:
            cfg.capture.source = int(args.source)
        except ValueError:
            cfg.capture.source = args.source

    if args.backend is not None:
        cfg.inference.backend = args.backend
    if args.model_path is not None:
        cfg.inference.model_path = args.model_path
    if args.model_runtime is not None:
        cfg.inference.model_runtime = args.model_runtime
    if args.device is not None:
        cfg.inference.device = args.device
    if args.model_channels is not None:
        cfg.inference.model_channels = args.model_channels
    if args.grpc_target is not None:
        cfg.inference.grpc_target = args.grpc_target
    if args.batch_size is not None:
        cfg.inference.batch_size = args.batch_size

    if args.stride is not None:
        cfg.preprocess.stride = args.stride

    if args.max_width is not None:
        cfg.display.max_display_width = args.max_width
    if args.alpha is not None:
        cfg.display.overlay_alpha = args.alpha
    if args.headless:
        cfg.display.headless = True


def main():
    args = parse_args()
    cfg = load_config()

    tprint("Loading streaming config...")
    if args.custom:
        tprint("Using default config.")
        apply_overrides(cfg, args)
    else:
        tprint("Using config/config.yaml.")

    tprint("Streaming pipeline configuration:")
    print(f"  Capture:    source={cfg.capture.source}, "
          f"{cfg.capture.width}x{cfg.capture.height} @ {cfg.capture.fps}fps")
    print(f"              profile={cfg.runtime.get('env_profile', 'server_test')}, "
          f"input_mode={cfg.capture.get('input_mode', 'opencv')}")
    if cfg.capture.get('input_mode') == 'npy_stream':
        print(f"              npy_dir={cfg.capture.get('npy_dir', '')}, "
              f"glob={cfg.capture.get('npy_glob', '*.npy')}, "
              f"npy_fps={cfg.capture.get('npy_fps', 10.0)}")
    print(f"  Preprocess: patch={cfg.preprocess.patch_size}, "
          f"stride={cfg.preprocess.stride}, mode={cfg.preprocess.normalize_mode}")
    print(f"  Inference:  backend={cfg.inference.backend}, "
          f"batch={cfg.inference.batch_size}")
    if cfg.inference.backend == "monai":
        print(f"              runtime={cfg.inference.model_runtime}, "
              f"device={cfg.inference.device}")
        print(f"              model={cfg.inference.model_path}, "
              f"channels={cfg.inference.model_channels}")
    elif cfg.inference.backend == "grpc":
        print(f"              target={cfg.inference.grpc_target}")
    else:
        print(f"              onnx={cfg.inference.model_path}")
    print(f"  Display:    {cfg.display.window_name}, "
            f"alpha={cfg.display.overlay_alpha}, "
            f"headless={cfg.display.get('headless', False)}")

    pipeline = StreamPipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()
