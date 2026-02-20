from __future__ import annotations


DEFAULT_CONFIG: dict = {
    "model": {
        "name": "communitycats",
        "path": None,
        "cfg_path": None,
        "weights_path": None,
        "labels_path": "yolo/cfg/custom-names-v4.txt",
        "confidence": 0.5,
        "nms": 0.5,
        "imgsz": 640,
        "class_filter": [],
    },
    "backend_policy": {
        "requested": "auto",
        "allow_darknet_fallback": True,
        "allow_rknn": True,
        "allow_tensorrt": True,
    },
    "ingest": {
        "uri": None,
        "backend": "auto",
        "queue_size": 2,
        "rate_limit_fps": None,
        "clock_mode": "auto",
        "benchmark": False,
        "gstreamer_pipeline": None,
        "pyav_options": {
            "rtsp_transport": "tcp",
            "fflags": "nobuffer",
            "flags": "low_delay",
            "max_delay": "500000",
        },
    },
    "output": {
        "headless": False,
        "window_name": "catDetectionSystem",
        "remote_enabled": False,
        "remote_host": "0.0.0.0",
        "remote_port": 8080,
        "remote_path": "/stream.mjpg",
    },
    "triggers": {
        "audio": {
            "enabled": True,
            "class_to_audio": {
                "cat": "assests/Meow-cat-sound-effect.mp3",
                "cat-domino": "assests/Meow-cat-sound-effect.mp3",
                "cat-olive": "assests/Meow-cat-sound-effect.mp3",
                "cat-bean": "assests/Meow-cat-sound-effect.mp3",
            },
            "cooldown_seconds": 15.0,
        },
        "hooks": {
            "enabled": False,
            "allowlist": [],
            "rules": [],
            "max_workers": 4,
        },
    },
    "monitoring": {
        "json_logs": False,
        "log_level": "INFO",
        "stats_interval_seconds": 5.0,
        "prometheus_enabled": False,
        "prometheus_host": "0.0.0.0",
        "prometheus_port": 9108,
        "event_stdout": True,
        "event_file": None,
    },
    "stress_sleep_ms": 0,
}
