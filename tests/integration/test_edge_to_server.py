"""
Integration tests for edge-to-server communication.
"""
import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch
import json

from edge.server_client import ServerClient
from edge.config import settings


class TestEdgeToServerIntegration:
    """Test edge device to server communication."""
    
    @pytest.fixture
    def server_client(self):
        return ServerClient()
    
    @pytest.fixture
    def sample_detection_event(self):
        return {
            "timestamp": 1642234567.89,
            "camera_id": 0,
            "detections": [
                {
                    "bbox": [100, 100, 200, 200],
                    "confidence": 0.85,
                    "class_id": 0,
                    "class_name": "person"
                }
            ]
        }
    
    @pytest.fixture
    def sample_evasion_event(self):
        return {
            "timestamp": 1642234567.89,
            "gate_id": "gate_001",
            "evasion_confidence": 0.85,
            "detection_events": [
                {
                    "timestamp": 1642234567.89,
                    "camera_id": 0,
                    "detections": [
                        {
                            "bbox": [100, 100, 200, 200],
                            "confidence": 0.85,
                            "class_id": 0,
                            "class_name": "person"
                        }
                    ]
                }
            ],
            "gate_event": {
                "timestamp": 1642234567.89,
                "gate_id": "gate_001",
                "is_open": True,
                "event_type": "open"
            },
            "event_id": "evasion_1642234567"
        }
    
    @pytest.mark.asyncio
    async def test_server_client_initialization(self, server_client):
        """Test server client initialization."""
        with patch('aiohttp.ClientSession') as mock_session:
            result = await server_client.initialize()
            
            assert result is True
            assert server_client.session is not None
            mock_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_detection_event_success(self, server_client, sample_detection_event):
        """Test successful detection event transmission."""
        mock_response = Mock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            await server_client.initialize()
            result = await server_client.send_detection_event(sample_detection_event)
            
            assert result is True
            mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_send_detection_event_failure(self, server_client, sample_detection_event):
        """Test detection event transmission failure."""
        mock_response = Mock()
        mock_response.status = 500
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            await server_client.initialize()
            result = await server_client.send_detection_event(sample_detection_event)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_send_evasion_event_with_snapshot(self, server_client, sample_evasion_event):
        """Test evasion event transmission with snapshot."""
        mock_response = Mock()
        mock_response.status = 200
        
        # Mock snapshot data
        snapshot_data = b"fake_image_data"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            await server_client.initialize()
            result = await server_client.send_evasion_event(sample_evasion_event, snapshot_data)
            
            assert result is True
            mock_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, server_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "healthy"}
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            await server_client.initialize()
            result = await server_client.health_check()
            
            assert result["status"] == "ok"
            assert "data" in result
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, server_client):
        """Test health check failure."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.get.side_effect = Exception("Connection failed")
            
            await server_client.initialize()
            result = await server_client.health_check()
            
            assert result["status"] == "error"
            assert "Connection failed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_client_cleanup(self, server_client):
        """Test client cleanup and resource management."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            
            await server_client.initialize()
            await server_client.close()
            
            mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, server_client, sample_detection_event):
        """Test handling of concurrent requests."""
        mock_response = Mock()
        mock_response.status = 200
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            await server_client.initialize()
            
            # Send multiple concurrent requests
            tasks = [
                server_client.send_detection_event(sample_detection_event)
                for _ in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # All requests should succeed
            assert all(results)
            assert mock_session.post.call_count == 5
