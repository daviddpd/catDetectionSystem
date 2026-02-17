from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DecoderProbeResult:
    selected_decoder: str
    reason: str
    available: list[str]


def _ffmpeg_hwaccels() -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        output = subprocess.check_output(
            [ffmpeg, "-hide_banner", "-hwaccels"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        )
    except Exception:
        return []

    lines = [line.strip().lower() for line in output.splitlines()]
    return [line for line in lines if line and not line.startswith("hardware")]


def probe_decoder_path() -> DecoderProbeResult:
    system = platform.system().lower()
    hwaccels = _ffmpeg_hwaccels()

    available = ["software"]
    available.extend(sorted(set(hwaccels)))

    if system == "darwin":
        if any("videotoolbox" in hw for hw in hwaccels):
            return DecoderProbeResult(
                selected_decoder="videotoolbox",
                reason="Apple platform with FFmpeg VideoToolbox support",
                available=available,
            )
        return DecoderProbeResult(
            selected_decoder="software",
            reason="Apple platform without FFmpeg VideoToolbox probe hit",
            available=available,
        )

    if system == "linux":
        if shutil.which("nvidia-smi") and any(
            hw in hwaccels for hw in ("cuda", "nvdec")
        ):
            return DecoderProbeResult(
                selected_decoder="nvdec",
                reason="NVIDIA GPU detected with FFmpeg CUDA/NVDEC support",
                available=available,
            )

        if any(Path(path).exists() for path in ("/dev/rknpu", "/dev/mpp_service")):
            return DecoderProbeResult(
                selected_decoder="rockchip-mpp",
                reason="Rockchip decode device nodes detected",
                available=available,
            )

        return DecoderProbeResult(
            selected_decoder="software",
            reason="No Linux hardware decoder path detected",
            available=available,
        )

    return DecoderProbeResult(
        selected_decoder="software",
        reason="Unknown platform, defaulting to software decode",
        available=available,
    )
