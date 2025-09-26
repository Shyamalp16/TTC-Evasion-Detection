"""
Unit tests for server API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from server.app.main import app
from server.app.models import Event, EventCreate


class TestServerAPI:
    """Test server API functionality."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def sample_event_data(self):
        return {
            "gate_id": "gate_001",
            "station_id": "station_001",
            "event_type": "detection",
            "camera_id": 0,
            "evasion_confidence": None,
            "num_detections": 2,
            "metadata": {
                "confidence_scores": [0.8, 0.9],
                "num_detections": 2
            }
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "SnitchSystem Server"
    
    @patch('server.app.main.EventService')
    def test_create_event(self, mock_service_class, client, sample_event_data):
        """Test event creation endpoint."""
        # Mock the service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock the created event
        mock_event = Mock()
        mock_event.id = 123
        mock_event.gate_id = "gate_001"
        mock_event.station_id = "station_001"
        mock_event.event_type = "detection"
        mock_event.camera_id = 0
        mock_event.evasion_confidence = None
        mock_event.num_detections = 2
        mock_event.metadata = {"confidence_scores": [0.8, 0.9]}
        mock_event.snapshot_path = None
        mock_event.created_at = "2024-01-15T10:30:01Z"
        mock_event.updated_at = "2024-01-15T10:30:01Z"
        
        mock_service.create_event.return_value = mock_event
        
        response = client.post("/events", json=sample_event_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 123
        assert data["gate_id"] == "gate_001"
        assert data["event_type"] == "detection"
    
    @patch('server.app.main.EventService')
    def test_get_events(self, mock_service_class, client):
        """Test events retrieval endpoint."""
        # Mock the service
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        # Mock events list
        mock_events = [
            Mock(id=1, gate_id="gate_001", event_type="detection"),
            Mock(id=2, gate_id="gate_001", event_type="evasion")
        ]
        mock_service.get_events.return_value = mock_events
        
        response = client.get("/events")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[1]["id"] == 2
    
    @patch('server.app.main.EventService')
    def test_get_events_with_filters(self, mock_service_class, client):
        """Test events retrieval with filters."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_events.return_value = []
        
        response = client.get("/events?event_type=detection&gate_id=gate_001&skip=0&limit=50")
        
        assert response.status_code == 200
        mock_service.get_events.assert_called_once_with(
            skip=0,
            limit=50,
            event_type="detection",
            gate_id="gate_001"
        )
    
    @patch('server.app.main.EventService')
    def test_get_event_by_id(self, mock_service_class, client):
        """Test getting specific event by ID."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_event = Mock()
        mock_event.id = 123
        mock_event.gate_id = "gate_001"
        mock_event.event_type = "detection"
        mock_service.get_event_by_id.return_value = mock_event
        
        response = client.get("/events/123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 123
        assert data["gate_id"] == "gate_001"
    
    @patch('server.app.main.EventService')
    def test_get_event_not_found(self, mock_service_class, client):
        """Test getting non-existent event."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.get_event_by_id.return_value = None
        
        response = client.get("/events/999")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('server.app.main.EventService')
    def test_get_statistics(self, mock_service_class, client):
        """Test statistics endpoint."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        
        mock_stats = {
            "total_events": 1000,
            "detection_events": 950,
            "evasion_events": 50,
            "events_today": 100,
            "events_this_week": 500,
            "events_this_month": 1000,
            "avg_evasion_confidence": 0.75,
            "most_active_gate": "gate_001",
            "recent_activity": []
        }
        mock_service.get_statistics.return_value = mock_stats
        
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_events"] == 1000
        assert data["detection_events"] == 950
        assert data["evasion_events"] == 50
    
    def test_create_event_with_snapshot(self, client):
        """Test event creation with snapshot upload."""
        # This would require more complex mocking for file upload
        # For now, just test that the endpoint exists
        response = client.post("/events/with-snapshot")
        
        # Should return 422 (validation error) due to missing form data
        assert response.status_code == 422
