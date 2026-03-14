# detect-planes-trains-automobiles

A Python application that watches a YouTube live stream (or any yt-dlp-compatible stream URL) and uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) to detect planes, trains, automobiles, trucks, and persons in each frame. Detection counts, average confidence scores, and additional metrics are published to an MQTT broker in real time.

## Features

- Streams video from any URL supported by [yt-dlp](https://github.com/yt-dlp/yt-dlp) (YouTube, etc.)
- Runs YOLO object detection on each frame, targeting COCO classes:
  - **Person** (class 0)
  - **Car** (class 2)
  - **Airplane** (class 4)
  - **Train** (class 6)
  - **Truck** (class 7)
- Automatically uses a CUDA GPU when available, falling back to CPU
- Publishes per-class detection counts and average confidence scores to MQTT
- Publishes aggregate metrics (total count, inference time, frames processed) to MQTT

## Requirements

- Python >= 3.14
- An MQTT broker (e.g. [Mosquitto](https://mosquitto.org/))
- A YOLO model file (e.g. `yolo11n.pt` downloaded from Ultralytics)

Python dependencies are declared in `pyproject.toml` and include `opencv-python`, `yt-dlp`, `paho-mqtt`, `ultralytics`, `torch`, and others.

## Installation

```bash
pip install .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/detect-pta.py [OPTIONS]
```

### Options

| Flag | Environment Variable | Default | Description |
|---|---|---|---|
| `--stream_url` | `STREAM_URL` | `https://www.youtube.com/watch?v=6dp-bvQ7RWo` | Stream or YouTube URL to analyse |
| `--mqtt_host` | `MQTT_HOST` | `localhost` | MQTT broker hostname |
| `--mqtt_port` | `MQTT_PORT` | `1883` | MQTT broker port |
| `--detection_model` | `DETECTION_MODEL` | `yolo26n.pt` | Path to the YOLO model file |

All options can be set via environment variables or command-line flags. Command-line flags take precedence.

### Example

```bash
# Using flags
python src/detect-pta.py \
  --stream_url "https://www.youtube.com/watch?v=<live-stream-id>" \
  --mqtt_host 192.168.1.10 \
  --mqtt_port 1883 \
  --detection_model yolo11n.pt

# Using environment variables
STREAM_URL="https://www.youtube.com/watch?v=<live-stream-id>" \
MQTT_HOST=192.168.1.10 \
DETECTION_MODEL=yolo11n.pt \
python src/detect-pta.py
```

## MQTT Topics

### Detection Counts

| Topic | Description |
|---|---|
| `pta/car/detection/count` | Number of cars detected in the current frame |
| `pta/person/detection/count` | Number of persons detected in the current frame |
| `pta/plane/detection/count` | Number of airplanes detected in the current frame |
| `pta/train/detection/count` | Number of trains detected in the current frame |
| `pta/truck/detection/count` | Number of trucks detected in the current frame |

### Average Confidence Scores

| Topic | Description |
|---|---|
| `pta/car/detection/confidence` | Average YOLO confidence for cars (0.0 – 1.0) |
| `pta/person/detection/confidence` | Average YOLO confidence for persons (0.0 – 1.0) |
| `pta/plane/detection/confidence` | Average YOLO confidence for airplanes (0.0 – 1.0) |
| `pta/train/detection/confidence` | Average YOLO confidence for trains (0.0 – 1.0) |
| `pta/truck/detection/confidence` | Average YOLO confidence for trucks (0.0 – 1.0) |

### Aggregate Metrics

| Topic | Description |
|---|---|
| `pta/detection/total/count` | Total detected objects in the current frame |
| `pta/detection/inference_time` | YOLO inference time in milliseconds |
| `pta/detection/frames_processed` | Cumulative number of frames processed |

## License

See [LICENSE](LICENSE).
