import sys
from unittest.mock import MagicMock

# Mock heavy dependencies so we can import publish_topics without them
for mod in ["cv2", "yt_dlp", "torch", "paho", "paho.mqtt", "paho.mqtt.client", "ultralytics"]:
    sys.modules.setdefault(mod, MagicMock())

from src.detect_pta import publish_topics


def test_publish_topics_publishes_all_counts():
    client = MagicMock()
    counts = {
        "pta/car/detection/count": 3,
        "pta/person/detection/count": 1,
    }
    confidences = {
        "pta/car/detection/confidence": 0.85,
    }
    publish_topics(client, counts, confidences, total_count=4, inference_time=12.345, frames_processed=1)

    # Verify all count topics were published
    client.publish.assert_any_call("pta/car/detection/count", "3")
    client.publish.assert_any_call("pta/person/detection/count", "1")

    # Verify confidence topic was published
    client.publish.assert_any_call("pta/car/detection/confidence", "0.85")

    # Verify additional metrics were published
    client.publish.assert_any_call("pta/detection/total/count", "4")
    client.publish.assert_any_call("pta/detection/inference_time", "12.35")
    client.publish.assert_any_call("pta/detection/frames_processed", "1")


def test_publish_topics_total_call_count():
    client = MagicMock()
    counts = {
        "pta/car/detection/count": 2,
        "pta/person/detection/count": 0,
        "pta/plane/detection/count": 0,
        "pta/train/detection/count": 0,
        "pta/truck/detection/count": 1,
    }
    confidences = {
        "pta/car/detection/confidence": 0.9,
        "pta/person/detection/confidence": 0.0,
        "pta/plane/detection/confidence": 0.0,
        "pta/train/detection/confidence": 0.0,
        "pta/truck/detection/confidence": 0.75,
    }
    publish_topics(client, counts, confidences, total_count=3, inference_time=50.0, frames_processed=10)

    # 5 counts + 5 confidences + 3 additional metrics = 13 publish calls
    assert client.publish.call_count == 13


def test_publish_topics_empty_detections():
    client = MagicMock()
    counts = {}
    confidences = {}
    publish_topics(client, counts, confidences, total_count=0, inference_time=5.0, frames_processed=1)

    # Only 3 additional metrics should be published
    assert client.publish.call_count == 3
    client.publish.assert_any_call("pta/detection/total/count", "0")
    client.publish.assert_any_call("pta/detection/inference_time", "5.0")
    client.publish.assert_any_call("pta/detection/frames_processed", "1")
