# System Architecture

## Overview

The SnitchSystem is a privacy-focused fare evasion detection system designed for subway gates. The system operates entirely at the station level with no cloud inference, ensuring passenger privacy while maintaining effective monitoring.

## System Components

### Edge Devices (Raspberry Pi/Jetson)
- **Location**: At each fare gate
- **Hardware**: 2 RTSP cameras per gate
- **Processing**: Local AI inference using YOLOv8-tiny
- **Privacy**: All processing happens on-device

### Local Station Server
- **Location**: At each subway station
- **Purpose**: Aggregate data from edge devices
- **Storage**: PostgreSQL database with local file storage
- **API**: FastAPI REST endpoints

## Data Flow

```
Cameras → Edge Device → Local Server → (Optional) Central Systems
```

1. **Detection**: Cameras capture video streams
2. **Processing**: Edge device runs person detection
3. **Correlation**: Compare detections with gate events
4. **Storage**: Send events to local server
5. **Aggregation**: Station-level statistics and reporting

## Privacy Architecture

### Data Minimization
- No face recognition or biometric data
- No personal identification
- Only detection events and metadata
- Automatic data retention policies

### Local Processing
- All AI inference on edge devices
- No cloud-based processing
- Local storage only
- Optional aggregate reporting to central systems

### Data Types
- **Detection Events**: Person detected, confidence, timestamp
- **Evasion Events**: Correlated detection + gate events
- **Snapshots**: Image captures for flagged events only
- **Metadata**: Gate ID, station ID, technical parameters

## Technology Stack

### Edge Devices
- **OS**: Linux (Raspberry Pi OS / Ubuntu)
- **Runtime**: Python 3.10+
- **AI**: YOLOv8-tiny (Ultralytics)
- **Vision**: OpenCV, GStreamer
- **Communication**: HTTP/HTTPS to local server

### Local Server
- **Runtime**: Python 3.10+
- **Framework**: FastAPI
- **Database**: PostgreSQL
- **Storage**: Local file system
- **Deployment**: Docker Compose

## Network Architecture

### Local Network
- Edge devices connect to station server via local network
- No external internet required for core functionality
- Optional VPN for remote monitoring

### Security
- API key authentication
- HTTPS for all communications
- Network isolation for edge devices
- Regular security updates

## Scalability

### Per Station
- Multiple gates per station
- Multiple cameras per gate
- Load balancing for high-traffic stations

### Multi-Station
- Independent station servers
- Optional central aggregation
- Distributed processing

## Monitoring and Maintenance

### Health Checks
- Edge device connectivity
- Camera stream status
- Server API health
- Database connectivity

### Logging
- Structured logging with Loguru
- Local log files
- Optional central log aggregation
- Privacy-compliant log retention

## Deployment Considerations

### Hardware Requirements
- **Edge**: Raspberry Pi 4+ or Jetson Nano
- **Server**: Standard x86 server or workstation
- **Storage**: SSD recommended for database
- **Network**: Gigabit Ethernet for camera streams

### Environmental
- Temperature and humidity considerations
- Power backup systems
- Network redundancy
- Physical security
