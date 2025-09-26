"""
Database models and Pydantic schemas.
"""
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.database import Base


class Event(Base):
    """Event database model."""
    __tablename__ = "events"
    
    id = Column(Integer, primary_key=True, index=True)
    gate_id = Column(String, index=True, nullable=False)
    station_id = Column(String, index=True, nullable=False)
    event_type = Column(String, index=True, nullable=False)  # "detection" or "evasion"
    timestamp = Column(DateTime, default=func.now(), index=True)
    
    # Detection/Evasion specific fields
    camera_id = Column(Integer, nullable=True)
    evasion_confidence = Column(Float, nullable=True)
    num_detections = Column(Integer, default=0)
    
    # Event metadata
    event_metadata = Column(JSON, nullable=True)
    snapshot_path = Column(String, nullable=True)
    
    # System fields
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class Gate(Base):
    """Gate configuration model."""
    __tablename__ = "gates"
    
    id = Column(Integer, primary_key=True, index=True)
    gate_id = Column(String, unique=True, index=True, nullable=False)
    station_id = Column(String, index=True, nullable=False)
    is_active = Column(Boolean, default=True)
    location = Column(String, nullable=True)
    camera_urls = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# Pydantic schemas for API
class EventBase(BaseModel):
    gate_id: str
    station_id: str
    event_type: str
    timestamp: Optional[datetime] = None
    camera_id: Optional[int] = None
    evasion_confidence: Optional[float] = None
    num_detections: Optional[int] = None
    event_metadata: Optional[Dict[str, Any]] = None


class EventCreate(EventBase):
    """Schema for creating events."""
    pass


class EventResponse(EventBase):
    """Schema for event responses."""
    id: int
    snapshot_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class EventListResponse(BaseModel):
    """Schema for event list responses."""
    events: List[EventResponse]
    total: int
    skip: int
    limit: int


class StatisticsResponse(BaseModel):
    """Schema for statistics responses."""
    total_events: int
    detection_events: int
    evasion_events: int
    events_today: int
    events_this_week: int
    events_this_month: int
    avg_evasion_confidence: Optional[float] = None
    most_active_gate: Optional[str] = None
    recent_activity: List[Dict[str, Any]]
