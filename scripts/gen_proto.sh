#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m grpc_tools.protoc \
  -I server/proto \
  --python_out=server \
  --grpc_python_out=server \
  server/proto/infer.proto

echo "Generated: server/infer_pb2.py server/infer_pb2_grpc.py"
