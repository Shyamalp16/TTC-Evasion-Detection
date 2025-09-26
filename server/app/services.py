"""
Business logic services.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from loguru import logger

from app.models import Event, EventCreate, StatisticsResponse


class EventService:
    """Service for event operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    async def create_event(self, event_data: EventCreate) -> Event:
        """Create a new event."""
        try:
            # Set timestamp if not provided
            if not event_data.timestamp:
                event_data.timestamp = datetime.now()
            
            # Create event record
            event = Event(
                gate_id=event_data.gate_id,
                station_id=event_data.station_id,
                event_type=event_data.event_type,
                timestamp=event_data.timestamp,
                camera_id=event_data.camera_id,
                evasion_confidence=event_data.evasion_confidence,
                num_detections=event_data.num_detections,
                event_metadata=event_data.event_metadata
            )
            
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            
            return event
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating event: {e}")
            raise
    
    async def create_event_with_snapshot(self, event_data: EventCreate, snapshot_path: str) -> Event:
        """Create an event with snapshot path."""
        try:
            event = await self.create_event(event_data)
            
            # Update event with snapshot path
            event.snapshot_path = snapshot_path
            self.db.commit()
            self.db.refresh(event)
            
            return event
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating event with snapshot: {e}")
            raise
    
    async def get_events(
        self,
        skip: int = 0,
        limit: int = 100,
        event_type: Optional[str] = None,
        gate_id: Optional[str] = None
    ) -> List[Event]:
        """Retrieve events with filtering."""
        try:
            query = self.db.query(Event)
            
            # Apply filters
            if event_type:
                query = query.filter(Event.event_type == event_type)
            if gate_id:
                query = query.filter(Event.gate_id == gate_id)
            
            # Apply pagination and ordering
            events = query.order_by(desc(Event.timestamp)).offset(skip).limit(limit).all()
            
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            raise
    
    async def get_event_by_id(self, event_id: int) -> Optional[Event]:
        """Get event by ID."""
        try:
            return self.db.query(Event).filter(Event.id == event_id).first()
        except Exception as e:
            logger.error(f"Error retrieving event {event_id}: {e}")
            raise
    
    async def get_statistics(self) -> StatisticsResponse:
        """Get station statistics."""
        try:
            now = datetime.now()
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # Total events
            total_events = self.db.query(Event).count()
            
            # Event type counts
            detection_events = self.db.query(Event).filter(Event.event_type == "detection").count()
            evasion_events = self.db.query(Event).filter(Event.event_type == "evasion").count()
            
            # Time-based counts
            events_today = self.db.query(Event).filter(
                func.date(Event.timestamp) == today
            ).count()
            
            events_this_week = self.db.query(Event).filter(
                Event.timestamp >= week_ago
            ).count()
            
            events_this_month = self.db.query(Event).filter(
                Event.timestamp >= month_ago
            ).count()
            
            # Average evasion confidence
            avg_confidence_result = self.db.query(
                func.avg(Event.evasion_confidence)
            ).filter(
                Event.event_type == "evasion",
                Event.evasion_confidence.isnot(None)
            ).scalar()
            
            avg_evasion_confidence = float(avg_confidence_result) if avg_confidence_result else None
            
            # Most active gate
            most_active_gate_result = self.db.query(
                Event.gate_id,
                func.count(Event.id).label('event_count')
            ).group_by(Event.gate_id).order_by(desc('event_count')).first()
            
            most_active_gate = most_active_gate_result[0] if most_active_gate_result else None
            
            # Recent activity (last 10 events)
            recent_events = self.db.query(Event).order_by(desc(Event.timestamp)).limit(10).all()
            recent_activity = [
                {
                    "id": event.id,
                    "gate_id": event.gate_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "evasion_confidence": event.evasion_confidence
                }
                for event in recent_events
            ]
            
            return StatisticsResponse(
                total_events=total_events,
                detection_events=detection_events,
                evasion_events=evasion_events,
                events_today=events_today,
                events_this_week=events_this_week,
                events_this_month=events_this_month,
                avg_evasion_confidence=avg_evasion_confidence,
                most_active_gate=most_active_gate,
                recent_activity=recent_activity
            )
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            raise
    
    async def cleanup_old_events(self, retention_days: int = 90) -> int:
        """Clean up old events based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Count events to be deleted
            old_events = self.db.query(Event).filter(Event.timestamp < cutoff_date)
            count = old_events.count()
            
            # Delete old events
            old_events.delete()
            self.db.commit()
            
            logger.info(f"Cleaned up {count} old events")
            return count
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error cleaning up old events: {e}")
            raise
