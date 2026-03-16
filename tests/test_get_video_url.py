import pytest
from unittest.mock import MagicMock, patch

from src.detect_pta import get_video_url


MOCK_URL = "https://example.com/stream.mp4"


def _make_ydl_mock(info_dict):
    """Return a patched yt_dlp.YoutubeDL context manager that yields the given info_dict."""
    ydl_instance = MagicMock()
    ydl_instance.extract_info.return_value = info_dict
    ydl_cm = MagicMock()
    ydl_cm.__enter__ = MagicMock(return_value=ydl_instance)
    ydl_cm.__exit__ = MagicMock(return_value=False)
    return ydl_cm, ydl_instance


class TestGetVideoUrl:
    def test_returns_url_for_best_mp4_under_1440p(self):
        """Returns the URL of the first mp4 format with height <= 1440."""
        formats = [
            {"ext": "mp4", "height": 720, "url": MOCK_URL},
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result == MOCK_URL

    def test_returns_first_suitable_format_when_multiple_exist(self):
        """When multiple mp4 formats qualify, the first one is returned."""
        first_url = "https://example.com/720.mp4"
        formats = [
            {"ext": "mp4", "height": 720, "url": first_url},
            {"ext": "mp4", "height": 1080, "url": "https://example.com/1080.mp4"},
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result == first_url

    def test_skips_formats_above_1440p(self):
        """Formats with height > 1440 must be ignored; returns None when nothing else qualifies."""
        formats = [
            {"ext": "mp4", "height": 2160, "url": "https://example.com/4k.mp4"},
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result is None

    def test_skips_non_mp4_formats(self):
        """Non-mp4 formats (e.g. webm) must be ignored."""
        formats = [
            {"ext": "webm", "height": 720, "url": "https://example.com/720.webm"},
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result is None

    def test_skips_formats_missing_url_key(self):
        """Formats without a 'url' key must be skipped."""
        formats = [
            {"ext": "mp4", "height": 720},  # no 'url'
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result is None

    def test_returns_none_when_formats_list_is_empty(self):
        """Returns None when the extracted info contains no formats."""
        ydl_cm, _ = _make_ydl_mock({"formats": []})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result is None

    def test_returns_none_on_extraction_exception(self):
        """Returns None (and does not raise) when yt_dlp raises an exception."""
        ydl_instance = MagicMock()
        ydl_instance.extract_info.side_effect = Exception("network error")
        ydl_cm = MagicMock()
        ydl_cm.__enter__ = MagicMock(return_value=ydl_instance)
        ydl_cm.__exit__ = MagicMock(return_value=False)

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result is None

    def test_format_with_missing_height_is_treated_as_zero(self):
        """A format without a 'height' key defaults to 0, which satisfies height <= 1440."""
        formats = [
            {"ext": "mp4", "url": MOCK_URL},  # no 'height'
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result == MOCK_URL

    def test_accepts_format_at_exactly_1440p(self):
        """A format at exactly 1440p height is included (boundary condition)."""
        formats = [
            {"ext": "mp4", "height": 1440, "url": MOCK_URL},
        ]
        ydl_cm, _ = _make_ydl_mock({"formats": formats})

        with patch("src.detect_pta.yt_dlp.YoutubeDL", return_value=ydl_cm):
            result = get_video_url("https://www.youtube.com/watch?v=dummy")

        assert result == MOCK_URL
