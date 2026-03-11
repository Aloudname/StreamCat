# pipeline/__init__.py — Public API for the real-time streaming pipeline.

from pipeline.packets import (FramePacket, InferPacket, ResultPacket)
from pipeline.capture import CaptureSource
from pipeline.preprocess import StreamPreprocessor
from pipeline.inference import (InferClient, TritonClient, OnnxClient,
                                  batched_infer)
from pipeline.postprocess import TileStitcher
from pipeline.display import StreamDisplay
from pipeline.core import (StreamPipeline, LatestQueue)
from pipeline.monitor import (LatencyTracker, StMonitor, tprint, RcrsMonitor)

__all__ = [
    # data packets
    'FramePacket',
    'InferPacket',
    'ResultPacket',

    # pipeline stages
    'CaptureSource',
    'StreamPreprocessor',
    'InferClient',
    'TritonClient',
    'OnnxClient',
    'batched_infer',
    'TileStitcher',
    'StreamDisplay',

    # orchestration
    'StreamPipeline',
    'LatestQueue',

    # monitoring
    'LatencyTracker',
    'StMonitor',
    'tprint',
    'RcrsMonitor'
]
