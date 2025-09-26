# Setup Guide

This guide will help you set up the SnitchSystem fare evasion detection system.

## Prerequisites

### Hardware Requirements

#### Edge Devices (Raspberry Pi/Jetson)
- Raspberry Pi 4+ (4GB RAM minimum) or Jetson Nano
- 2x IP cameras with RTSP support
- MicroSD card (32GB+ for Raspberry Pi)
- Power supply and network connectivity

#### Local Station Server
- x86_64 server or workstation
- 8GB+ RAM
- 100GB+ storage (SSD recommended)
- Network connectivity to edge devices

### Software Requirements
- Python 3.10+
- Docker and Docker Compose (for server)
- PostgreSQL (if not using Docker)
- Git

## Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SnitchSystem
```

### 2. Edge Device Setup

#### Install Dependencies
```bash
cd edge
pip install -r requirements.txt
```

#### Configure Environment
```bash
cp env.example .env
# Edit .env with your camera URLs and server settings
```

#### Configure Cameras
Update the camera URLs in your `.env` file:
```bash
SNITCH_CAMERA_URLS=["rtsp://admin:password@192.168.1.100:554/stream1","rtsp://admin:password@192.168.1.101:554/stream1"]
```

#### Run Edge Device
```bash
python main.py
```

### 3. Server Setup

#### Option A: Docker Compose (Recommended)
```bash
cd server
docker-compose up -d
```

#### Option B: Manual Setup
```bash
cd server
pip install -r requirements.txt

# Configure PostgreSQL
# Update database connection in .env

# Run database migrations
python -c "from app.database import create_tables; create_tables()"

# Start server
python run.py
```

#### Configure Environment
```bash
cp env.example .env
# Edit .env with your database and API settings
```

## Detailed Configuration

### Edge Device Configuration

#### Camera Setup
1. Configure your IP cameras to stream RTSP
2. Test camera streams:
   ```bash
   ffplay rtsp://admin:password@192.168.1.100:554/stream1
   ```
3. Update camera URLs in `edge/.env`

#### Detection Settings
- `SNITCH_DETECTION_CONFIDENCE`: Detection confidence threshold (0.0-1.0)
- `SNITCH_DETECTION_IOU_THRESHOLD`: IoU threshold for NMS (0.0-1.0)
- `SNITCH_FRAME_SKIP`: Process every Nth frame (performance tuning)

#### Server Connection
- `SNITCH_SERVER_URL`: Local server API endpoint
- `SNITCH_SERVER_API_KEY`: Authentication key

### Server Configuration

#### Database Setup
```bash
# Create database
createdb snitchsystem

# Create user
createuser snitch
psql -c "ALTER USER snitch PASSWORD 'password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE snitchsystem TO snitch;"
```

#### API Configuration
- `SNITCH_API_KEY`: API key for edge device authentication
- `SNITCH_SECRET_KEY`: Secret key for JWT tokens
- `SNITCH_DATABASE_URL`: PostgreSQL connection string

#### File Storage
- `SNITCH_SNAPSHOTS_DIR`: Directory for storing snapshots
- `SNITCH_MAX_FILE_SIZE_MB`: Maximum snapshot file size

## Testing

### Run Unit Tests
```bash
# Edge device tests
cd edge
pytest tests/unit/

# Server tests
cd server
pytest tests/unit/
```

### Run Integration Tests
```bash
# Start test server
cd server
docker-compose up -d

# Run integration tests
cd tests
pytest integration/
```

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Create test event
curl -X POST http://localhost:8000/events \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key-here" \
  -d '{
    "gate_id": "gate_001",
    "station_id": "station_001",
    "event_type": "detection",
    "camera_id": 0,
    "num_detections": 1,
    "metadata": {"confidence_scores": [0.85]}
  }'
```

## Monitoring and Maintenance

### Health Checks
- Edge device: Check logs in `edge/logs/`
- Server: Visit `http://localhost:8000/health`
- Database: Check PostgreSQL logs

### Log Files
- Edge: `edge/logs/edge_device.log`
- Server: `server/logs/server.log`
- Database: PostgreSQL logs

### Performance Monitoring
- Monitor CPU usage on edge devices
- Check memory usage for YOLO model
- Monitor database performance
- Check network connectivity

### Data Cleanup
```bash
# Clean old events (run periodically)
cd server
python -c "
from app.database import SessionLocal
from app.services import EventService
db = SessionLocal()
service = EventService(db)
service.cleanup_old_events(90)  # 90 days retention
"
```

## Troubleshooting

### Common Issues

#### Edge Device
1. **Camera connection failed**
   - Check camera URLs and credentials
   - Verify network connectivity
   - Test with VLC or ffplay

2. **YOLO model not loading**
   - Check internet connection for model download
   - Verify PyTorch installation
   - Check available memory

3. **Server connection failed**
   - Verify server URL and port
   - Check API key configuration
   - Test network connectivity

#### Server
1. **Database connection failed**
   - Check PostgreSQL service status
   - Verify connection string
   - Check database permissions

2. **API authentication failed**
   - Verify API key configuration
   - Check Authorization header format
   - Review server logs

### Debug Mode
```bash
# Edge device debug
SNITCH_LOG_LEVEL=DEBUG python main.py

# Server debug
SNITCH_LOG_LEVEL=DEBUG python run.py
```

## Security Considerations

### Network Security
- Use VPN for remote access
- Implement firewall rules
- Use HTTPS for all communications
- Regular security updates

### Data Privacy
- Review data retention policies
- Implement access controls
- Regular security audits
- Monitor for unauthorized access

### API Security
- Use strong API keys
- Implement rate limiting
- Enable HTTPS
- Regular key rotation

## Production Deployment

### Edge Devices
1. Use production-ready hardware
2. Implement monitoring and alerting
3. Set up automatic updates
4. Configure backup systems

### Server Infrastructure
1. Use load balancers for high availability
2. Implement database clustering
3. Set up monitoring and alerting
4. Configure backup and disaster recovery

### Scaling
1. Multiple gates per station
2. Load balancing for high traffic
3. Database sharding for large deployments
4. Central aggregation for multi-station deployments
