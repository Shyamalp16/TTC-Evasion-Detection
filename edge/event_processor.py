"""
Event processing and correlation logic.
"""
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
from config import settings


@dataclass
class DetectionEvent:
    """Detection event data structure."""
    timestamp: float
    camera_id: int
    detections: List[Dict[str, Any]]
    frame_data: Optional[bytes] = None


@dataclass
class GateEvent:
    """Gate state event data structure."""
    timestamp: float
    gate_id: str
    is_open: bool
    event_type: str  # "open", "close", "blocked"


class EventProcessor:
    """Processes detection events and correlates with gate state."""
    
    def __init__(self):
        self.detection_events = []
        self.gate_events = []
        self.evasion_threshold = 2.0  # seconds
        self.max_detection_age = 10.0  # seconds
        self.pending_detections = []  # batch regular detections
        self.last_batch_sent = time.time()
        
    async def add_detection_event(self, camera_id: int, detections: List[Dict[str, Any]], 
                                frame_data: Optional[bytes] = None) -> DetectionEvent:
        """Add a new detection event."""
        event = DetectionEvent(
            timestamp=time.time(),
            camera_id=camera_id,
            detections=detections,
            frame_data=frame_data
        )
        
        self.detection_events.append(event)
        
        # Add to pending batch for regular detections
        self.pending_detections.append({
            "timestamp": event.timestamp,
            "camera_id": camera_id,
            "detections": detections
        })
        
        logger.debug(f"Detection event added: camera {camera_id}, {len(detections)} detections")
        
        # Clean old events
        await self._cleanup_old_events()
        
        return event
    
    async def add_gate_event(self, gate_id: str, is_open: bool, event_type: str) -> GateEvent:
        """Add a gate state event."""
        event = GateEvent(
            timestamp=time.time(),
            gate_id=gate_id,
            is_open=is_open,
            event_type=event_type
        )
        
        self.gate_events.append(event)
        logger.info(f"Gate event: {gate_id} {event_type}")
        
        # Check for potential evasion
        await self._check_for_evasion(event)
        
        return event
    
    async def _check_for_evasion(self, gate_event: GateEvent) -> Optional[Dict[str, Any]]:
        """Check if gate event correlates with detection events indicating evasion."""
        if not gate_event.is_open:
            return None  # Only check when gate opens
        
        current_time = gate_event.timestamp
        
        # Find detection events within the evasion threshold
        relevant_detections = []
        for detection in self.detection_events:
            time_diff = abs(detection.timestamp - current_time)
            if time_diff <= self.evasion_threshold:
                relevant_detections.append(detection)
        
        if not relevant_detections:
            return None
        
        # Analyze detection patterns
        evasion_confidence = await self._calculate_evasion_confidence(relevant_detections)
        
        if evasion_confidence > 0.7:  # Threshold for flagging
            evasion_event = {
                "timestamp": current_time,
                "gate_id": gate_event.gate_id,
                "evasion_confidence": evasion_confidence,
                "detection_events": relevant_detections,
                "gate_event": gate_event,
                "event_id": f"evasion_{int(current_time)}"
            }
            
            logger.warning(f"Potential fare evasion detected: {evasion_event['event_id']}")
            return evasion_event
        
        return None
    
    async def get_pending_detections_batch(self) -> List[Dict[str, Any]]:
        """Get pending detections for batching."""
        if not self.pending_detections:
            return []
        
        # Check if it's time to send a batch
        current_time = time.time()
        if current_time - self.last_batch_sent < settings.detection_batch_interval:
            return []
        
        # Return pending detections and clear the list
        batch = self.pending_detections.copy()
        self.pending_detections.clear()
        self.last_batch_sent = current_time
        
        logger.info(f"Prepared detection batch: {len(batch)} events")
        return batch
    
    async def _calculate_evasion_confidence(self, detections: List[DetectionEvent]) -> float:
        """Calculate confidence score for evasion detection."""
        if not detections:
            return 0.0
        
        # Factors for evasion confidence:
        # 1. Number of detection events
        # 2. Number of cameras with detections
        # 3. Detection confidence scores
        # 4. Temporal proximity to gate event
        
        num_events = len(detections)
        num_cameras = len(set(d.camera_id for d in detections))
        
        total_confidence = 0.0
        total_detections = 0
        
        for detection in detections:
            for det in detection.detections:
                total_confidence += det["confidence"]
                total_detections += 1
        
        avg_confidence = total_confidence / total_detections if total_detections > 0 else 0.0
        
        # Simple scoring algorithm
        confidence = min(1.0, (num_events * 0.3 + num_cameras * 0.3 + avg_confidence * 0.4))
        
        return confidence
    
    async def _cleanup_old_events(self):
        """Remove old events to prevent memory buildup."""
        current_time = time.time()
        
        # Clean detection events
        self.detection_events = [
            event for event in self.detection_events
            if current_time - event.timestamp < self.max_detection_age
        ]
        
        # Clean gate events (keep more history)
        self.gate_events = [
            event for event in self.gate_events
            if current_time - event.timestamp < self.max_detection_age * 2
        ]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        current_time = time.time()
        
        recent_detections = len([
            event for event in self.detection_events
            if current_time - event.timestamp < 60  # Last minute
        ])
        
        recent_gate_events = len([
            event for event in self.gate_events
            if current_time - event.timestamp < 60
        ])
        
        return {
            "total_detection_events": len(self.detection_events),
            "total_gate_events": len(self.gate_events),
            "recent_detections_1min": recent_detections,
            "recent_gate_events_1min": recent_gate_events,
            "evasion_threshold": self.evasion_threshold,
            "max_detection_age": self.max_detection_age
        }
