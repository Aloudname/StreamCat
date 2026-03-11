#!/usr/bin/env python3
"""
run.py — Entry point for the real-time streaming inference pipeline.

Usage examples:
    # Default config from config/config.yaml
    python run.py

    # Custom config by arg cmd.
    python run.py --custom/-c

    # Override: use Triton backend
    python run.py -c --backend/-b triton --triton-url/-url localhost:8001

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

    # Inference overrides
    parser.add_argument('--backend', '-b', type=str, default=None,
                        choices=['triton', 'onnx'],
                        help='Inference backend (default: from config)')
    parser.add_argument('--triton-url', '-url', type=str, default=None,
                        help='Triton server gRPC URL (e.g. localhost:8001)')
    parser.add_argument('--onnx-path', '-p', type=str, default=None,
                        help='Path to ONNX model file')
    parser.add_argument('--model-name', '-n', type=str, default=None,
                        help='Triton model name')
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

    return parser.parse_args()


def apply_overrides(cfg, args):
    """Override config values from CLI arguments."""
    if args.source:
        # Try to parse as int (device index), otherwise treat as file path
        try:
            cfg.capture.source = int(args.source)
        except ValueError:
            cfg.capture.source = args.source

    if args.backend is not None:
        cfg.inference.backend = args.backend
    if args.triton_url is not None:
        cfg.inference.triton_url = args.triton_url
    if args.onnx_path is not None:
        cfg.inference.onnx_path = args.onnx_path
    if args.model_name is not None:
        cfg.inference.model_name = args.model_name
    if args.batch_size is not None:
        cfg.inference.batch_size = args.batch_size

    if args.stride is not None:
        cfg.preprocess.stride = args.stride

    if args.max_width is not None:
        cfg.display.max_display_width = args.max_width
    if args.alpha is not None:
        cfg.display.overlay_alpha = args.alpha


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
    print(f"  Preprocess: patch={cfg.preprocess.patch_size}, "
          f"stride={cfg.preprocess.stride}, mode={cfg.preprocess.normalize_mode}")
    print(f"  Inference:  backend={cfg.inference.backend}, "
          f"batch={cfg.inference.batch_size}")
    if cfg.inference.backend == "triton":
        print(f"              url={cfg.inference.triton_url}, "
              f"model={cfg.inference.model_name}")
    else:
        print(f"              onnx={cfg.inference.onnx_path}")
    print(f"  Display:    {cfg.display.window_name}, "
          f"alpha={cfg.display.overlay_alpha}")

    pipeline = StreamPipeline(cfg)
    pipeline.run()


if __name__ == '__main__':
    main()
