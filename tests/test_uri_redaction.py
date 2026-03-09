from __future__ import annotations

import unittest

from cds.utils import redact_uri_password


class UriRedactionTests(unittest.TestCase):
    def test_redacts_password_in_rtsp_uri(self) -> None:
        source = "rtsp://camera-user:super-secret@example.local:554/live/main"
        self.assertEqual(
            redact_uri_password(source),
            "rtsp://camera-user:***@example.local:554/live/main",
        )

    def test_preserves_query_and_fragment(self) -> None:
        source = "http://user:pw@example.local/path?q=1#fragment"
        self.assertEqual(
            redact_uri_password(source),
            "http://user:***@example.local/path?q=1#fragment",
        )

    def test_leaves_user_only_uri_unchanged(self) -> None:
        source = "rtsp://camera-user@example.local:554/live/main"
        self.assertEqual(redact_uri_password(source), source)

    def test_leaves_non_uri_source_unchanged(self) -> None:
        source = "/Volumes/camera/communitycats/referenceVideos/file.mp4"
        self.assertEqual(redact_uri_password(source), source)


if __name__ == "__main__":
    unittest.main()
