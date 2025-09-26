# Local Station Server

This module contains the FastAPI server and database that runs at each subway station.

## Purpose

- Receive detection events from edge devices
- Store event data and snapshots in PostgreSQL
- Provide API endpoints for data retrieval
- Aggregate statistics for station management
- Handle data export to central systems (if needed)

## Key Components

- **FastAPI Application**: REST API for event ingestion
- **PostgreSQL Database**: Event storage and metadata
- **Image Storage**: Local file system for snapshots
- **Data Models**: SQLAlchemy models for event tracking
- **Authentication**: Basic API key authentication

## Setup Requirements

- Python 3.10+
- FastAPI for web API
- SQLAlchemy for database ORM
- PostgreSQL database
- Docker Compose for easy deployment

## Installation

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Manual Setup

```bash
pip install -r requirements.txt
# Configure PostgreSQL connection
# Run database migrations
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /events` - Receive detection events from edge devices
- `GET /events` - Retrieve event history
- `GET /stats` - Station statistics
- `GET /health` - Health check

## Database Schema

- **events**: Detection events with timestamps and metadata
- **snapshots**: Image file references and metadata
- **gates**: Gate configuration and status

## Configuration

Edit `config.py` for:
- Database connection settings
- API authentication keys
- File storage paths
- Logging configuration

## Privacy & Data Handling

- All data remains at the station level
- No personal information is stored
- Only aggregate statistics may be exported
- Images are stored locally with automatic cleanup
