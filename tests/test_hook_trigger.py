from __future__ import annotations

import unittest
from unittest.mock import patch

from cds.config.models import HookRuleConfig, HookTriggerConfig
from cds.triggers.hooks import HookTrigger


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return None

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        return


class HookTriggerTests(unittest.TestCase):
    def test_command_template_renders_detection_metadata(self) -> None:
        trigger = HookTrigger(
            HookTriggerConfig(
                enabled=True,
                allowlist=["/usr/local/bin/cds-hook"],
                rules=[
                    HookRuleConfig(
                        classes=["cat"],
                        command=[
                            "/usr/local/bin/cds-hook",
                            "{label}",
                            "{class_id}",
                            "{confidence:.2f}",
                            "{frame_id}",
                            "{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2}",
                        ],
                        cooldown_seconds=0.0,
                    )
                ],
            )
        )
        trigger._executor = _ImmediateExecutor()  # type: ignore[assignment]
        payload = {
            "label": "cat",
            "class_id": 3,
            "confidence": 0.876,
            "frame_id": 42,
            "bbox": [11, 22, 33, 44],
        }

        with patch("subprocess.run") as run_mock:
            trigger.emit("cat", payload)

        run_mock.assert_called_once()
        command = run_mock.call_args.args[0]
        self.assertEqual(
            command,
            [
                "/usr/local/bin/cds-hook",
                "cat",
                "3",
                "0.88",
                "42",
                "11,22,33,44",
            ],
        )
        trigger.close()

    def test_env_payload_exports_flattened_metadata(self) -> None:
        trigger = HookTrigger(
            HookTriggerConfig(
                enabled=True,
                allowlist=["/usr/local/bin/cds-hook"],
                rules=[
                    HookRuleConfig(
                        classes=["cat"],
                        command=["/usr/local/bin/cds-hook", "{label}"],
                        cooldown_seconds=0.0,
                        payload_mode="env",
                    )
                ],
            )
        )
        trigger._executor = _ImmediateExecutor()  # type: ignore[assignment]
        payload = {
            "ts": "2026-04-03T12:00:00+00:00",
            "label": "cat",
            "class_id": 3,
            "confidence": 0.876,
            "frame_id": 42,
            "source": "rtsp://camera-user:***@example.local/live",
            "backend": "rknn",
            "bbox": [11, 22, 33, 44],
            "area_pixels": 999,
            "area_percent": 1.5,
        }

        with patch("subprocess.run") as run_mock:
            trigger.emit("cat", payload)

        run_mock.assert_called_once()
        env = run_mock.call_args.kwargs["env"]
        self.assertEqual(env["CDS_LABEL"], "cat")
        self.assertEqual(env["CDS_CLASS_ID"], "3")
        self.assertEqual(env["CDS_FRAME_ID"], "42")
        self.assertEqual(env["CDS_BBOX_X1"], "11")
        self.assertEqual(env["CDS_BBOX_Y2"], "44")
        self.assertEqual(env["CDS_SOURCE"], "rtsp://camera-user:***@example.local/live")
        self.assertIn("CDS_EVENT_JSON", env)
        trigger.close()
