# Edge Device Module

This module contains the code that runs on Raspberry Pi or Jetson devices at each fare gate.

## Purpose

- Connect to 2 RTSP cameras per gate
- Run YOLOv8-tiny person detection in real-time
- Compare detection events with gate open/close events
- Flag potential fare evasion incidents
- Capture snapshots when evasion is detected
- Send event data to local station server

## Key Components

- **Camera Interface**: RTSP stream handling for dual camera setup
- **AI Detection**: YOLOv8-tiny person detection pipeline
- **Event Correlation**: Match detection events with gate state
- **Image Capture**: Snapshot generation for flagged events
- **Data Transmission**: Send events to local server API

## Setup Requirements

- Python 3.10+
- OpenCV for camera handling
- PyTorch/TensorFlow Lite for AI inference
- Ultralytics YOLO for object detection
- GStreamer for RTSP camera input
- Network access to local station server

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set:
- Camera RTSP URLs
- Server API endpoint
- Detection confidence thresholds
- Gate ID and location

## Usage

```bash
python main.py
```

## Privacy Notes

- All processing happens locally on the edge device
- No data is sent to external cloud services
- Only event metadata and snapshots are transmitted to local server
- No face recognition or personal identification
