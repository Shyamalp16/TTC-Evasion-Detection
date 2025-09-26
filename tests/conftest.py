"""
Pytest configuration and shared fixtures.
"""
import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from server.app.database import Base, get_db
from server.app.models import Event


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    # Create in-memory SQLite database for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    return engine, TestingSessionLocal, override_get_db


@pytest.fixture
def sample_event_data():
    """Sample event data for testing."""
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


@pytest.fixture
def sample_evasion_data():
    """Sample evasion event data for testing."""
    return {
        "gate_id": "gate_001",
        "station_id": "station_001",
        "event_type": "evasion",
        "camera_id": 0,
        "evasion_confidence": 0.85,
        "num_detections": 2,
        "metadata": {
            "confidence_scores": [0.8, 0.9],
            "num_detections": 2,
            "evasion_confidence": 0.85
        }
    }


@pytest.fixture
def mock_camera_frame():
    """Mock camera frame for testing."""
    import numpy as np
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection_results():
    """Mock YOLO detection results for testing."""
    return [
        {
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class_id": 0,
            "class_name": "person"
        },
        {
            "bbox": [300, 300, 400, 400],
            "confidence": 0.75,
            "class_id": 0,
            "class_name": "person"
        }
    ]


@pytest.fixture
def temp_snapshots_dir():
    """Create temporary directory for snapshots."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for testing."""
    mock_model = Mock()
    mock_result = Mock()
    mock_box = Mock()
    
    # Configure mock box
    mock_box.xyxy = [Mock()]
    mock_box.conf = [Mock()]
    mock_box.cls = [Mock()]
    
    mock_result.boxes = [mock_box]
    mock_model.return_value = [mock_result]
    
    return mock_model


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test environment variables
    os.environ["SNITCH_LOG_LEVEL"] = "DEBUG"
    os.environ["SNITCH_GATE_ID"] = "test_gate"
    os.environ["SNITCH_STATION_ID"] = "test_station"
    
    yield
    
    # Cleanup after test
    pass
