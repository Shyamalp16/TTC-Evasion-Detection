"""
FastAPI application main module.
"""
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
from loguru import logger

from app.database import get_db
from app.models import Event, EventCreate, EventResponse
from app.services import EventService
from app.config import settings

# Create FastAPI application
app = FastAPI(
    title="SnitchSystem Server API",
    description="Local station server for fare evasion detection system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "SnitchSystem Server",
        "version": "1.0.0"
    }


@app.post("/events", response_model=EventResponse)
async def create_event(
    event_data: EventCreate,
    db=Depends(get_db)
):
    """Create a new detection or evasion event."""
    try:
        event_service = EventService(db)
        event = await event_service.create_event(event_data)
        
        logger.info(f"Event created: {event.id} - {event.event_type}")
        
        return EventResponse.from_orm(event)
        
    except Exception as e:
        logger.error(f"Error creating event: {e}")
        raise HTTPException(status_code=500, detail="Failed to create event")


@app.post("/events/with-snapshot")
async def create_event_with_snapshot(
    event_data: str = Form(...),
    snapshot: UploadFile = File(...),
    db=Depends(get_db)
):
    """Create an event with image snapshot."""
    try:
        import json
        from app.models import EventCreate
        
        # Parse event data
        event_dict = json.loads(event_data)
        event_create = EventCreate(**event_dict)
        
        # Save snapshot
        snapshot_path = await save_snapshot(snapshot)
        
        # Create event with snapshot path
        event_service = EventService(db)
        event = await event_service.create_event_with_snapshot(event_create, snapshot_path)
        
        logger.info(f"Event with snapshot created: {event.id}")
        
        return {"message": "Event created successfully", "event_id": event.id}
        
    except Exception as e:
        logger.error(f"Error creating event with snapshot: {e}")
        raise HTTPException(status_code=500, detail="Failed to create event with snapshot")


@app.get("/events", response_model=List[EventResponse])
async def get_events(
    skip: int = 0,
    limit: int = 100,
    event_type: Optional[str] = None,
    gate_id: Optional[str] = None,
    db=Depends(get_db)
):
    """Retrieve events with optional filtering."""
    try:
        event_service = EventService(db)
        events = await event_service.get_events(
            skip=skip,
            limit=limit,
            event_type=event_type,
            gate_id=gate_id
        )
        
        return [EventResponse.from_orm(event) for event in events]
        
    except Exception as e:
        logger.error(f"Error retrieving events: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve events")


@app.get("/events/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, db=Depends(get_db)):
    """Get a specific event by ID."""
    try:
        event_service = EventService(db)
        event = await event_service.get_event_by_id(event_id)
        
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        
        return EventResponse.from_orm(event)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving event {event_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve event")


@app.get("/stats")
async def get_statistics(db=Depends(get_db)):
    """Get station statistics."""
    try:
        event_service = EventService(db)
        stats = await event_service.get_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


async def save_snapshot(file: UploadFile) -> str:
    """Save uploaded snapshot file."""
    import os
    import uuid
    from datetime import datetime
    
    # Create snapshots directory if it doesn't exist
    snapshots_dir = "snapshots"
    os.makedirs(snapshots_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"snapshot_{timestamp}_{unique_id}.jpg"
    file_path = os.path.join(snapshots_dir, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    return file_path


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
