"""Tests for the detect-planes-trains-automobiles package."""

from detect_planes_trains_automobiles import __version__
from detect_planes_trains_automobiles.__main__ import main


def test_version():
    assert __version__ == "0.1.0"


def test_main(capsys):
    main()
    captured = capsys.readouterr()
    assert "detect-planes-trains-automobiles" in captured.out
