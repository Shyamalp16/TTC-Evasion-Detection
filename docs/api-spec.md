# API Specification

## Base URL
```
http://localhost:8000
```

## Authentication
All API endpoints require authentication via API key in the Authorization header:
```
Authorization: Bearer your-api-key-here
```

## Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "SnitchSystem Server",
  "version": "1.0.0"
}
```

### Create Event
```http
POST /events
Content-Type: application/json
```

**Request Body:**
```json
{
  "gate_id": "gate_001",
  "station_id": "station_001",
  "event_type": "detection",
  "timestamp": "2024-01-15T10:30:00Z",
  "camera_id": 0,
  "evasion_confidence": 0.85,
  "num_detections": 2,
  "metadata": {
    "confidence_scores": [0.92, 0.78],
    "num_detections": 2
  }
}
```

**Response:**
```json
{
  "id": 123,
  "gate_id": "gate_001",
  "station_id": "station_001",
  "event_type": "detection",
  "timestamp": "2024-01-15T10:30:00Z",
  "camera_id": 0,
  "evasion_confidence": 0.85,
  "num_detections": 2,
  "metadata": {
    "confidence_scores": [0.92, 0.78],
    "num_detections": 2
  },
  "snapshot_path": null,
  "created_at": "2024-01-15T10:30:01Z",
  "updated_at": "2024-01-15T10:30:01Z"
}
```

### Create Event with Snapshot
```http
POST /events/with-snapshot
Content-Type: multipart/form-data
```

**Form Data:**
- `event_data`: JSON string of event data
- `snapshot`: Image file (JPEG/PNG)

### Get Events
```http
GET /events?skip=0&limit=100&event_type=detection&gate_id=gate_001
```

**Query Parameters:**
- `skip`: Number of events to skip (default: 0)
- `limit`: Maximum number of events to return (default: 100)
- `event_type`: Filter by event type ("detection" or "evasion")
- `gate_id`: Filter by gate ID

**Response:**
```json
[
  {
    "id": 123,
    "gate_id": "gate_001",
    "station_id": "station_001",
    "event_type": "detection",
    "timestamp": "2024-01-15T10:30:00Z",
    "camera_id": 0,
    "evasion_confidence": null,
    "num_detections": 2,
    "metadata": {...},
    "snapshot_path": null,
    "created_at": "2024-01-15T10:30:01Z",
    "updated_at": "2024-01-15T10:30:01Z"
  }
]
```

### Get Event by ID
```http
GET /events/{event_id}
```

**Response:**
```json
{
  "id": 123,
  "gate_id": "gate_001",
  "station_id": "station_001",
  "event_type": "evasion",
  "timestamp": "2024-01-15T10:30:00Z",
  "camera_id": 0,
  "evasion_confidence": 0.85,
  "num_detections": 2,
  "metadata": {...},
  "snapshot_path": "/snapshots/snapshot_20240115_103000_abc123.jpg",
  "created_at": "2024-01-15T10:30:01Z",
  "updated_at": "2024-01-15T10:30:01Z"
}
```

### Get Statistics
```http
GET /stats
```

**Response:**
```json
{
  "total_events": 15420,
  "detection_events": 15200,
  "evasion_events": 220,
  "events_today": 450,
  "events_this_week": 3200,
  "events_this_month": 15420,
  "avg_evasion_confidence": 0.78,
  "most_active_gate": "gate_003",
  "recent_activity": [
    {
      "id": 123,
      "gate_id": "gate_001",
      "event_type": "evasion",
      "timestamp": "2024-01-15T10:30:00Z",
      "evasion_confidence": 0.85
    }
  ]
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request data"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid API key"
}
```

### 404 Not Found
```json
{
  "detail": "Event not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error"
}
```

## Rate Limiting

- **API Endpoints**: 10 requests per second per IP
- **Burst**: 20 requests allowed
- **Headers**: Rate limit information in response headers

## Data Models

### Event Types
- `detection`: Person detected by camera
- `evasion`: Potential fare evasion (correlated detection + gate event)

### Metadata Fields
- `confidence_scores`: Array of detection confidence values
- `num_detections`: Total number of detections in event
- `camera_urls`: Source camera URLs (for debugging)
- `processing_time`: Time taken for detection processing

### Timestamps
- All timestamps are in ISO 8601 format (UTC)
- `timestamp`: Event occurrence time
- `created_at`: Database record creation time
- `updated_at`: Last record update time
