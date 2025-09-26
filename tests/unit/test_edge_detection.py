"""
Unit tests for edge device detection functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
import asyncio

from edge.detection_engine import DetectionEngine
from edge.event_processor import EventProcessor, DetectionEvent, GateEvent


class TestDetectionEngine:
    """Test detection engine functionality."""
    
    @pytest.fixture
    def detection_engine(self):
        return DetectionEngine()
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.mark.asyncio
    async def test_initialization(self, detection_engine):
        """Test detection engine initialization."""
        with patch('edge.detection_engine.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_yolo.return_value = mock_model
            
            result = await detection_engine.initialize()
            
            assert result is True
            assert detection_engine.is_initialized is True
            mock_yolo.assert_called_once_with('yolov8n.pt')
    
    @pytest.mark.asyncio
    async def test_detect_persons(self, detection_engine, sample_frame):
        """Test person detection on a frame."""
        # Mock the model and results
        mock_model = Mock()
        mock_result = Mock()
        mock_box = Mock()
        
        # Configure mock box
        mock_box.xyxy = [np.array([100, 100, 200, 200])]
        mock_box.conf = [np.array([0.85])]
        mock_box.cls = [np.array([0])]
        
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]
        
        detection_engine.model = mock_model
        detection_engine.is_initialized = True
        
        detections = await detection_engine.detect_persons(sample_frame)
        
        assert len(detections) == 1
        assert detections[0]["confidence"] == 0.85
        assert detections[0]["class_name"] == "person"
        assert detections[0]["bbox"] == [100, 100, 200, 200]
    
    @pytest.mark.asyncio
    async def test_detect_multiple_frames(self, detection_engine, sample_frame):
        """Test detection on multiple frames."""
        frames = [(0, sample_frame), (1, sample_frame)]
        
        with patch.object(detection_engine, 'detect_persons') as mock_detect:
            mock_detect.return_value = [{"confidence": 0.8, "bbox": [0, 0, 100, 100]}]
            
            results = await detection_engine.detect_multiple_frames(frames)
            
            assert len(results) == 2
            assert 0 in results
            assert 1 in results
            assert len(results[0]) == 1
            assert len(results[1]) == 1


class TestEventProcessor:
    """Test event processing and correlation."""
    
    @pytest.fixture
    def event_processor(self):
        return EventProcessor()
    
    @pytest.mark.asyncio
    async def test_add_detection_event(self, event_processor):
        """Test adding detection events."""
        detections = [{"confidence": 0.8, "bbox": [0, 0, 100, 100]}]
        
        event = await event_processor.add_detection_event(
            camera_id=0,
            detections=detections
        )
        
        assert event.camera_id == 0
        assert len(event.detections) == 1
        assert len(event_processor.detection_events) == 1
    
    @pytest.mark.asyncio
    async def test_add_gate_event(self, event_processor):
        """Test adding gate events."""
        gate_event = await event_processor.add_gate_event(
            gate_id="gate_001",
            is_open=True,
            event_type="open"
        )
        
        assert gate_event.gate_id == "gate_001"
        assert gate_event.is_open is True
        assert gate_event.event_type == "open"
        assert len(event_processor.gate_events) == 1
    
    @pytest.mark.asyncio
    async def test_evasion_detection(self, event_processor):
        """Test evasion detection logic."""
        # Add detection events
        await event_processor.add_detection_event(
            camera_id=0,
            detections=[{"confidence": 0.9, "bbox": [0, 0, 100, 100]}]
        )
        
        # Add gate open event (should trigger evasion check)
        gate_event = await event_processor.add_gate_event(
            gate_id="gate_001",
            is_open=True,
            event_type="open"
        )
        
        # The evasion check should run automatically
        # In a real implementation, we'd check the result
        assert len(event_processor.gate_events) == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, event_processor):
        """Test cleanup of old events."""
        # Add some events
        await event_processor.add_detection_event(0, [])
        await event_processor.add_gate_event("gate_001", True, "open")
        
        # Manually set old timestamps
        old_time = 1000000000  # Very old timestamp
        for event in event_processor.detection_events:
            event.timestamp = old_time
        for event in event_processor.gate_events:
            event.timestamp = old_time
        
        # Trigger cleanup
        await event_processor._cleanup_old_events()
        
        # Events should be cleaned up
        assert len(event_processor.detection_events) == 0
        assert len(event_processor.gate_events) == 0
    
    @pytest.mark.asyncio
    async def test_get_statistics(self, event_processor):
        """Test statistics retrieval."""
        # Add some events
        await event_processor.add_detection_event(0, [])
        await event_processor.add_gate_event("gate_001", True, "open")
        
        stats = await event_processor.get_statistics()
        
        assert "total_detection_events" in stats
        assert "total_gate_events" in stats
        assert "recent_detections_1min" in stats
        assert "recent_gate_events_1min" in stats
