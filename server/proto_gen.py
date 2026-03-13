import importlib
import pathlib
import sys


def ensure_proto_generated() -> None:
    root = pathlib.Path(__file__).resolve().parent
    proto_dir = root / "proto"
    pb2 = root / "infer_pb2.py"
    pb2_grpc = root / "infer_pb2_grpc.py"

    if pb2.exists() and pb2_grpc.exists():
        return

    try:
        from grpc_tools import protoc
    except ImportError as exc:
        raise ImportError("grpcio-tools is required to generate gRPC stubs") from exc

    args = [
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={root}",
        f"--grpc_python_out={root}",
        str(proto_dir / "infer.proto"),
    ]
    ret = protoc.main(args)
    if ret != 0:
        raise RuntimeError(f"protoc failed with exit code {ret}")

    importlib.invalidate_caches()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
