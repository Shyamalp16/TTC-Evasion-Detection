"""
Client for communicating with the local station server.
"""
import asyncio
import json
import time
import aiohttp
from typing import Dict, Any, Optional, List
from loguru import logger
from config import settings


class ServerClient:
    """HTTP client for communicating with the station server."""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.base_url = settings.server_url
        self.api_key = settings.server_api_key
        self.timeout = aiohttp.ClientTimeout(total=settings.server_timeout)
        
    async def initialize(self) -> bool:
        """Initialize HTTP session."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
            logger.info("Server client initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize server client: {e}")
            return False
    
    async def send_detection_event(self, event_data: Dict[str, Any]) -> bool:
        """Send detection event to server."""
        if not self.session:
            logger.error("Server client not initialized")
            return False
        
        try:
            url = f"{self.base_url}/events"
            
            payload = {
                "gate_id": settings.gate_id,
                "station_id": settings.station_id,
                "event_type": "detection",
                "timestamp": event_data.get("timestamp"),
                "camera_id": event_data.get("camera_id"),
                "detections": event_data.get("detections", []),
                "metadata": {
                    "confidence_scores": [d.get("confidence", 0) for d in event_data.get("detections", [])],
                    "num_detections": len(event_data.get("detections", []))
                }
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug(f"Detection event sent successfully")
                    return True
                else:
                    logger.error(f"Failed to send detection event: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending detection event: {e}")
            return False
    
    async def send_detection_batch(self, batch_data: List[Dict[str, Any]]) -> bool:
        """Send batched detection events to server."""
        if not self.session:
            logger.error("Server client not initialized")
            return False
        
        try:
            # Send each detection as individual event (fallback since server doesn't have batch endpoint)
            success_count = 0
            for event_data in batch_data:
                try:
                    payload = {
                        "gate_id": settings.gate_id,
                        "station_id": settings.station_id,
                        "event_type": "detection",
                        "timestamp": event_data["timestamp"],
                        "camera_id": event_data["camera_id"],
                        "detections": event_data["detections"],
                        "event_metadata": {
                            "confidence_scores": [d.get("confidence", 0) for d in event_data["detections"]],
                            "num_detections": len(event_data["detections"])
                        }
                    }
                    
                    async with self.session.post(f"{self.base_url}/events", json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                        else:
                            logger.warning(f"Failed to send individual detection: {response.status}")
                            
                except Exception as e:
                    logger.warning(f"Error sending individual detection: {e}")
            
            if success_count > 0:
                logger.info(f"Detection batch sent: {success_count}/{len(batch_data)} events successful")
                return True
            else:
                logger.error(f"Failed to send any detections from batch of {len(batch_data)}")
                return False
                    
        except Exception as e:
            logger.error(f"Error sending detection batch: {e}")
            return False

    async def send_evasion_event(self, evasion_data: Dict[str, Any], snapshot_data: Optional[bytes] = None) -> bool:
        """Send evasion event with optional snapshot to server."""
        if not self.session:
            logger.error("Server client not initialized")
            return False
        
        try:
            url = f"{self.base_url}/events"
            
            # Convert DetectionEvent objects to dictionaries for JSON serialization
            detection_events_dict = []
            for detection_event in evasion_data.get("detection_events", []):
                detection_events_dict.append({
                    "timestamp": detection_event.timestamp,
                    "camera_id": detection_event.camera_id,
                    "detections": detection_event.detections
                })
            
            # Convert GateEvent object to dictionary
            gate_event = evasion_data.get("gate_event")
            gate_event_dict = {
                "timestamp": gate_event.timestamp,
                "gate_id": gate_event.gate_id,
                "is_open": gate_event.is_open,
                "event_type": gate_event.event_type
            } if gate_event else None
            
            payload = {
                "gate_id": settings.gate_id,
                "station_id": settings.station_id,
                "event_type": "evasion",
                "timestamp": evasion_data.get("timestamp"),
                "evasion_confidence": evasion_data.get("evasion_confidence"),
                "detection_events": detection_events_dict,
                "gate_event": gate_event_dict,
                "event_id": evasion_data.get("event_id"),
                "event_metadata": {
                    "num_detection_events": len(detection_events_dict),
                    "total_detections": sum(len(de["detections"]) for de in detection_events_dict),
                    "evasion_confidence": evasion_data.get("evasion_confidence")
                }
            }
            
            # If snapshot data is provided, send as multipart
            if snapshot_data:
                data = aiohttp.FormData()
                data.add_field('event_data', json.dumps(payload), content_type='application/json')
                data.add_field('snapshot', snapshot_data, filename='snapshot.jpg', content_type='image/jpeg')
                
                async with self.session.post(url, data=data) as response:
                    if response.status == 200:
                        logger.warning(f"Evasion event with snapshot sent: {evasion_data.get('event_id')}")
                        return True
                    else:
                        logger.error(f"Failed to send evasion event: {response.status}")
                        return False
            else:
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.warning(f"Evasion event sent: {evasion_data.get('event_id')}")
                        return True
                    else:
                        logger.error(f"Failed to send evasion event: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error sending evasion event: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server connectivity."""
        if not self.session:
            return {"status": "error", "message": "Client not initialized"}
        
        try:
            url = f"{self.base_url}/health"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "ok", "data": data}
                else:
                    return {"status": "error", "message": f"HTTP {response.status}"}
                    
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def close(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            logger.info("Server client closed")
