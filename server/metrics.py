from prometheus_client import Counter, Gauge, Histogram, start_http_server


class ServiceMetrics:
    def __init__(self) -> None:
        self.requests_total = Counter(
            "streamcat_requests_total",
            "Total inference requests",
            ["protocol", "status"],
        )
        self.infer_latency_ms = Histogram(
            "streamcat_infer_latency_ms",
            "Inference latency in milliseconds",
            buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
        )
        self.infer_batch_size = Histogram(
            "streamcat_infer_batch_size",
            "Inference batch size",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
        )
        self.ready_state = Gauge(
            "streamcat_ready",
            "Readiness status",
        )

    def start_metrics_server(self, port: int) -> None:
        start_http_server(port)
