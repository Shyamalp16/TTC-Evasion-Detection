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
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            logger.info("Server client initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize server client: {e}")
            return False
    
    def _build_event_payload(self, event_data: Dict[str, Any], default_event_type: Optional[str] = None) -> Dict[str, Any]:
        """Normalize arbitrary event_data into server's EventCreate schema payload."""
        from datetime import datetime, timezone

        payload: Dict[str, Any] = {
            "gate_id": settings.gate_id,
            "station_id": settings.station_id,
            "event_type": event_data.get("event_type") or default_event_type or "detection",
        }

        ts = event_data.get("timestamp")
        if isinstance(ts, (int, float)):
            try:
                payload["timestamp"] = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
            except Exception:
                # If conversion fails, omit timestamp so server sets it
                pass
        elif isinstance(ts, str) and ts:
            payload["timestamp"] = ts

        camera_id = event_data.get("camera_id")
        if isinstance(camera_id, int):
            payload["camera_id"] = camera_id

        ev_conf = event_data.get("evasion_confidence")
        if isinstance(ev_conf, (int, float)):
            payload["evasion_confidence"] = float(ev_conf)

        # num_detections from provided value or derived from detections list
        num_detections = event_data.get("num_detections")
        if not isinstance(num_detections, int):
            dets = event_data.get("detections")
            if isinstance(dets, list):
                num_detections = len(dets)
        if isinstance(num_detections, int):
            payload["num_detections"] = num_detections

        # Merge metadata
        metadata: Dict[str, Any] = {}
        if isinstance(event_data.get("event_metadata"), dict):
            metadata.update(event_data["event_metadata"])  # type: ignore[arg-type]

        # Pull through commonly used extra fields into metadata bucket
        for key in ("detections", "detection_events", "gate_event", "confidence_scores", "crossing_direction", "detection_method", "event_id"):
            value = event_data.get(key)
            if value is not None:
                metadata[key] = value

        if metadata:
            payload["event_metadata"] = metadata

        return payload
    
    async def send_detection_event(self, event_data: Dict[str, Any]) -> bool:
        """Send detection event to server."""
        if not self.session:
            logger.error("Server client not initialized")
            return False
        
        try:
            url = f"{self.base_url}/events"
            
            payload = self._build_event_payload({
                **event_data,
                "event_type": "detection",
                "num_detections": len(event_data.get("detections", [])),
                "confidence_scores": [d.get("confidence", 0) for d in event_data.get("detections", [])],
            }, default_event_type="detection")
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.debug(f"Detection event sent successfully")
                    return True
                else:
                    try:
                        error_text = await response.text()
                    except Exception:
                        error_text = "<no body>"
                    logger.error(f"Failed to send detection event: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error sending detection event: {e}")
            return False
    
    async def send_event(self, event_data: Dict[str, Any], snapshot_data: Optional[bytes] = None) -> bool:
        """Send a single event to the server."""
        if not self.session:
            logger.error("Server client not initialized")
            return False

        try:
            if snapshot_data:
                # Use the with-snapshot endpoint
                url = f"{self.base_url}/events/with-snapshot"

                # Convert event_data to JSON string for form data
                import json
                payload = self._build_event_payload(event_data)
                event_data_json = json.dumps(payload)

                # Create multipart form data
                data = aiohttp.FormData()

                # Add the JSON data as a form field
                data.add_field('event_data', event_data_json)

                # Add the image data as a file field
                data.add_field('snapshot', snapshot_data, filename='evasion_snapshot.jpg', content_type='image/jpeg')

                async with self.session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        event_id = result.get('event_id') or result.get('id')
                        logger.warning(f"Event with snapshot sent: {event_id}")
                        # Verify snapshot persisted on server (non-blocking)
                        if event_id is not None:
                            asyncio.create_task(self._verify_snapshot_on_server(int(event_id)))
                        return True
                    else:
                        # Log the response content for debugging
                        try:
                            error_content = await response.text()
                            logger.error(f"Failed to send event with snapshot: {response.status} - {error_content}")
                        except Exception as e:
                            logger.error(f"Failed to send event with snapshot: {response.status} - Error reading response: {e}")
                        return False
            else:
                # Use the regular events endpoint
                url = f"{self.base_url}/events"
                payload = self._build_event_payload(event_data)
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Event sent: {result.get('id', 'unknown')}")
                        return True
                    else:
                        try:
                            error_text = await response.text()
                        except Exception:
                            error_text = "<no body>"
                        logger.error(f"Failed to send event: {response.status} - {error_text}")
                        return False

        except Exception as e:
            logger.error(f"Error sending event: {e}")
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
                    payload = self._build_event_payload({
                        "event_type": "detection",
                        "timestamp": event_data.get("timestamp"),
                        "camera_id": event_data.get("camera_id"),
                        "detections": event_data.get("detections", []),
                        "num_detections": len(event_data.get("detections", [])),
                        "confidence_scores": [d.get("confidence", 0) for d in event_data.get("detections", [])],
                    }, default_event_type="detection")
                    
                    async with self.session.post(f"{self.base_url}/events", json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                        else:
                            try:
                                error_text = await response.text()
                            except Exception:
                                error_text = "<no body>"
                            logger.warning(f"Failed to send individual detection: {response.status} - {error_text}")
                            
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
            # Normalize evasion payload
            payload = self._build_event_payload({
                **evasion_data,
                "event_type": "evasion",
            }, default_event_type="evasion")

            # If snapshot data is provided, send as multipart to the with-snapshot endpoint
            if snapshot_data:
                url = f"{self.base_url}/events/with-snapshot"
                data = aiohttp.FormData()
                import json
                data.add_field('event_data', json.dumps(payload))
                data.add_field('snapshot', snapshot_data, filename='snapshot.jpg', content_type='image/jpeg')

                async with self.session.post(url, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        event_id = result.get('event_id') or result.get('id')
                        logger.warning(f"Evasion event with snapshot sent: {event_id}")
                        if event_id is not None:
                            asyncio.create_task(self._verify_snapshot_on_server(int(event_id)))
                        return True
                    else:
                        try:
                            error_text = await response.text()
                        except Exception:
                            error_text = "<no body>"
                        logger.error(f"Failed to send evasion event: {response.status} - {error_text}")
                        return False
            else:
                url = f"{self.base_url}/events"
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.warning(f"Evasion event sent: {evasion_data.get('event_id')}")
                        return True
                    else:
                        try:
                            error_text = await response.text()
                        except Exception:
                            error_text = "<no body>"
                        logger.error(f"Failed to send evasion event: {response.status} - {error_text}")
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

    async def _verify_snapshot_on_server(self, event_id: int) -> None:
        """Fetch event and log snapshot_path for verification."""
        if not self.session:
            return
        try:
            url = f"{self.base_url}/events/{event_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    snapshot_path = data.get('snapshot_path')
                    logger.info(f"Verified event {event_id}, snapshot_path: {snapshot_path}")
                else:
                    logger.warning(f"Could not verify snapshot for event {event_id}: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"Snapshot verification failed for event {event_id}: {e}")
