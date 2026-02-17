from __future__ import annotations

from cds.config.models import IngestConfig
from cds.io.ingest.base import VideoIngest


def _pyav_available() -> bool:
    try:
        import av

        _ = av.__version__
        return True
    except Exception:
        return False


def select_ingest_backend(config: IngestConfig) -> tuple[VideoIngest, str]:
    backend = config.backend.lower().strip()

    if backend == "pyav":
        from cds.io.ingest.pyav_ingest import PyAVIngest

        return PyAVIngest(), "Requested ingest backend pyav"

    if backend == "gstreamer":
        from cds.io.ingest.gstreamer_ingest import GStreamerIngest

        return GStreamerIngest(), "Requested ingest backend gstreamer"

    if backend == "opencv":
        from cds.io.ingest.opencv_ingest import OpenCVIngest

        return OpenCVIngest(), "Requested ingest backend opencv"

    # auto selection
    if _pyav_available():
        from cds.io.ingest.pyav_ingest import PyAVIngest

        return PyAVIngest(), "Auto ingest policy selected pyav"

    if config.gstreamer_pipeline:
        from cds.io.ingest.gstreamer_ingest import GStreamerIngest

        return GStreamerIngest(), "Auto ingest policy selected gstreamer pipeline"

    from cds.io.ingest.opencv_ingest import OpenCVIngest

    return OpenCVIngest(), "Auto ingest policy selected opencv fallback"
