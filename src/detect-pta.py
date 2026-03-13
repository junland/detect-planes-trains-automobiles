import os
import time
import argparse
import cv2
import yt_dlp
import torch
import paho.mqtt.client as mqtt
from ultralytics import YOLO

def get_video_url(stream_url):
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Extracting stream URL from: {stream_url}")
            info_dict = ydl.extract_info(stream_url, download=False)
            # Find the best that is has a max resolution of 1440p and is in mp4 format
            best_format = next((f for f in info_dict['formats'] if 'url' in f and f.get('ext') == 'mp4' and f.get('height', 0) <= 1440), None)
            if best_format is None:
                print("No suitable mp4 format with URL found in video info.")
                return None
            return best_format['url']
        except Exception as e:
            print(f"Error extracting stream URL: {e}")
            return None
        
def main():
    # Define environment variable support for argparse
    default_stream_url = os.getenv("STREAM_URL", "https://www.youtube.com/watch?v=6dp-bvQ7RWo")
    default_mqtt_host = os.getenv("MQTT_HOST", "localhost")
    default_mqtt_port = int(os.getenv("MQTT_PORT", 1883))
    default_detection_model = os.getenv("DETECTION_MODEL", "yolo26n.pt")

    # Define arguments
    parser = argparse.ArgumentParser(description="Detection for planes, trains, and automobiles (cars, trucks, etc.) from YouTube stream")
    parser.add_argument("--stream_url", type=str, default=default_stream_url, help="Stream / Youtube URL")
    parser.add_argument("--mqtt_host", type=str, default=default_mqtt_host, help="MQTT broker host")
    parser.add_argument("--mqtt_port", type=int, default=default_mqtt_port, help="MQTT broker port")
    parser.add_argument("--detection_model", type=str, default=default_detection_model, help="YOLO model file")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client_host = args.mqtt_host
    client_port = args.mqtt_port
    # Set up the model to use the GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.detection_model)
    stream_url = args.stream_url

    # Define MQTT topics for each class
    topic_cars_count = "car/detection/count"
    topic_person_count = "person/detection/count"
    topic_planes_count = "plane/detection/count"
    topic_trains_count = "train/detection/count"
    topic_trucks_count = "truck/detection/count"

    # Define MQTT topics for confidence scores
    topic_cars_confidence = "car/detection/confidence"
    topic_person_confidence = "person/detection/confidence"
    topic_planes_confidence = "plane/detection/confidence"
    topic_trains_confidence = "train/detection/confidence"
    topic_trucks_confidence = "truck/detection/confidence"

    # Define MQTT topics for additional metrics
    topic_total_count = "detection/total/count"
    topic_inference_time = "detection/inference_time"
    topic_frames_processed = "detection/frames_processed"

    # Print information about the device being used for inference
    print(f"Using device: {device} for inference.")

    # Make sure we can connect to mqtt broker before proceeding
    try:
        client.connect(client_host, client_port, 60)
        client.loop_start()
        print("Connected to MQTT broker...")
    except Exception as e:
        print(f"Error connecting to MQTT broker: {e}")
    
    cap = None
    
    try:
        print(f"Retrieving video URL from stream URL: {stream_url}")

        video_url = get_video_url(stream_url)
        
        if not video_url:
            print("Could not retrieve stream URL.")
            return None, 1
        
        print(f"Found video URL: {video_url}")
        
        print("Starting video capture...")

        cap = cv2.VideoCapture(video_url)
        
        if not cap.isOpened():
            print("Error opening video stream.")
            return None, 1
        
        frames_processed = 0

        while True:
            ret, frame = cap.read()

            print("Processing frame...")
            
            if not ret:
                print("Failed to read frame from stream.")
                break
            
            start_time = time.time()
            results = model(frame, verbose=False, device=device)
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            boxes = results[0].boxes

            # COCO class IDs: 0=person, 2=car, 4=airplane, 6=train, 7=truck
            cls = boxes.cls.tolist() if boxes is not None else []
            conf = boxes.conf.tolist() if boxes is not None else []
            
            num_cars   = cls.count(2)
            num_person = cls.count(0)
            num_planes = cls.count(4)
            num_trains = cls.count(6)
            num_trucks = cls.count(7)

            # Calculate average confidence per class
            class_ids = {2: [], 0: [], 4: [], 6: [], 7: []}
            for c, cf in zip(cls, conf):
                if c in class_ids:
                    class_ids[c].append(cf)

            avg_conf_cars   = sum(class_ids[2]) / len(class_ids[2]) if class_ids[2] else 0.0
            avg_conf_person = sum(class_ids[0]) / len(class_ids[0]) if class_ids[0] else 0.0
            avg_conf_planes = sum(class_ids[4]) / len(class_ids[4]) if class_ids[4] else 0.0
            avg_conf_trains = sum(class_ids[6]) / len(class_ids[6]) if class_ids[6] else 0.0
            avg_conf_trucks = sum(class_ids[7]) / len(class_ids[7]) if class_ids[7] else 0.0

            total_count = num_cars + num_person + num_planes + num_trains + num_trucks
            frames_processed += 1

            counts = {
                topic_cars_count:   num_cars,
                topic_person_count: num_person,
                topic_planes_count: num_planes,
                topic_trains_count: num_trains,
                topic_trucks_count: num_trucks, 
            }

            confidences = {
                topic_cars_confidence:   round(avg_conf_cars, 4),
                topic_person_confidence: round(avg_conf_person, 4),
                topic_planes_confidence: round(avg_conf_planes, 4),
                topic_trains_confidence: round(avg_conf_trains, 4),
                topic_trucks_confidence: round(avg_conf_trucks, 4),
            }

            for topic, count in counts.items():
                print(f"Publishing {count} to topic {topic}")
                payload_msg = f"{count}"
                client.publish(topic, payload_msg)

            for topic, confidence in confidences.items():
                print(f"Publishing {confidence} to topic {topic}")
                payload_msg = f"{confidence}"
                client.publish(topic, payload_msg)

            client.publish(topic_total_count, f"{total_count}")
            print(f"Publishing {total_count} to topic {topic_total_count}")
            client.publish(topic_inference_time, f"{round(inference_time, 2)}")
            print(f"Publishing {round(inference_time, 2)} to topic {topic_inference_time}")
            client.publish(topic_frames_processed, f"{frames_processed}")
            print(f"Publishing {frames_processed} to topic {topic_frames_processed}")

            print(f"Sent detection counts — cars: {num_cars}, trucks: {num_trucks}, planes: {num_planes}, trains: {num_trains}, persons: {num_person}")
            print(f"Sent detection confidences — cars: {avg_conf_cars:.4f}, trucks: {avg_conf_trucks:.4f}, planes: {avg_conf_planes:.4f}, trains: {avg_conf_trains:.4f}, persons: {avg_conf_person:.4f}")
            print(f"Sent additional metrics — total: {total_count}, inference_time: {inference_time:.2f}ms, frames_processed: {frames_processed}")
        
        return 0, None
    finally:
        if cap is not None:
            cap.release()

        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    _, err = main()

    if err:
        print(f"Error: {err}")
        exit(1)
