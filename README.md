# SnitchSystem - Fare Evasion Detection System

A privacy-focused fare evasion detection system for subway gates, designed for Toronto TTC-style installations.

## System Overview

This system monitors subway fare gates using computer vision to detect potential fare evasion while maintaining strict privacy requirements:

- **Edge Processing**: All AI inference runs locally on Raspberry Pi/Jetson devices
- **Dual Camera Setup**: Each gate has 2 cameras for comprehensive monitoring
- **YOLOv8-tiny**: Lightweight object detection for person detection
- **Local Server**: Station-level data aggregation and storage
- **Privacy First**: No cloud inference, no face recognition, no profiling

## Project Structure

```
SnitchSystem/
├── edge/           # Edge device code (Raspberry Pi/Jetson)
├── server/         # Local station server (FastAPI + PostgreSQL)
├── docs/           # Architecture diagrams and documentation
├── tests/          # Unit and integration tests
└── README.md       # This file
```

## Quick Start

1. **Edge Device Setup**: See `edge/README.md` for camera setup and YOLO installation
2. **Server Setup**: See `server/README.md` for FastAPI and database configuration
3. **Documentation**: Check `docs/` for system architecture and design decisions

## Privacy & Compliance

- All detection runs locally on-device
- No personal data collection or storage
- Only aggregate logs may be sent to central servers
- No face recognition or biometric data
- Event metadata only (timestamp, gate ID, detection confidence)

## Technology Stack

- **Edge**: Python 3.10, OpenCV, YOLOv8-tiny, GStreamer
- **Server**: FastAPI, SQLAlchemy, PostgreSQL, Docker
- **Cameras**: RTSP streams from IP cameras
- **AI**: Ultralytics YOLO for person detection

## License

[Add your license here]
