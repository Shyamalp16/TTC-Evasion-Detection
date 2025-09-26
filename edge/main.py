"""
Main application entry point for the edge device.
"""
import asyncio
import cv2
import time
from typing import List, Tuple
from loguru import logger
from config import settings

from camera_handler import CameraHandler
from detection_engine import DetectionEngine
from event_processor import EventProcessor
from server_client import ServerClient


class EdgeDevice:
    """Main edge device application."""
    
    def __init__(self):
        self.camera_handler = CameraHandler(settings.camera_urls)
        self.detection_engine = DetectionEngine()
        self.event_processor = EventProcessor()
        self.server_client = ServerClient()
        self.is_running = False
        self.frame_count = 0
        
    async def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("Initializing edge device...")
        
        # Initialize camera handler
        if not await self.camera_handler.initialize_cameras():
            logger.error("Failed to initialize cameras")
            return False
        
        # Initialize detection engine
        if not await self.detection_engine.initialize():
            logger.error("Failed to initialize detection engine")
            return False
        
        # Initialize server client
        if not await self.server_client.initialize():
            logger.error("Failed to initialize server client")
            return False
        
        logger.info("Edge device initialized successfully")
        return True
    
    async def run_detection_loop(self):
        """Main detection loop."""
        logger.info("Starting detection loop...")
        self.is_running = True
        
        while self.is_running:
            try:
                # Read frames from all cameras
                frames = await self.camera_handler.read_all_frames()
                
                if not frames:
                    logger.warning("No frames received from cameras")
                    await asyncio.sleep(0.1)
                    continue
                
                # Process frames for detection
                if self.frame_count % settings.frame_skip == 0:
                    await self._process_frames(frames)
                
                self.frame_count += 1
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(settings.detection_interval)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_frames(self, frames: List[Tuple[int, any]]):
        """Process frames for person detection."""
        try:
            # Run detection on all frames
            detection_results = await self.detection_engine.detect_multiple_frames(frames)
            
            # Process each camera's detections
            for camera_id, detections in detection_results.items():
                if detections:
                    logger.debug(f"Camera {camera_id}: {len(detections)} detections")
                    
                    # Add detection event
                    detection_event = await self.event_processor.add_detection_event(
                        camera_id=camera_id,
                        detections=detections
                    )
                    
                    # Send to server (async, don't wait)
                    asyncio.create_task(
                        self.server_client.send_detection_event({
                            "timestamp": detection_event.timestamp,
                            "camera_id": camera_id,
                            "detections": detections
                        })
                    )
            
            # Simulate gate events for testing (replace with actual gate integration)
            await self._simulate_gate_events()
            
        except Exception as e:
            logger.error(f"Error processing frames: {e}")
    
    async def _simulate_gate_events(self):
        """Simulate gate open/close events for testing."""
        # This is a placeholder - replace with actual gate integration
        import random
        
        if random.random() < 0.01:  # 1% chance per frame
            gate_event = await self.event_processor.add_gate_event(
                gate_id=settings.gate_id,
                is_open=True,
                event_type="open"
            )
            
            # Check for evasion after a short delay
            await asyncio.sleep(0.5)
            
            await self.event_processor.add_gate_event(
                gate_id=settings.gate_id,
                is_open=False,
                event_type="close"
            )
    
    async def health_check(self) -> dict:
        """Perform health check on all components."""
        health_status = {
            "timestamp": time.time(),
            "gate_id": settings.gate_id,
            "station_id": settings.station_id,
            "is_running": self.is_running,
            "frame_count": self.frame_count
        }
        
        # Camera health
        camera_health = await self.camera_handler.health_check()
        health_status["cameras"] = camera_health
        
        # Detection engine health
        model_info = await self.detection_engine.get_model_info()
        health_status["detection_engine"] = model_info
        
        # Event processor statistics
        event_stats = await self.event_processor.get_statistics()
        health_status["event_processor"] = event_stats
        
        # Server connectivity
        server_health = await self.server_client.health_check()
        health_status["server"] = server_health
        
        return health_status
    
    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down edge device...")
        self.is_running = False
        
        # Release camera resources
        self.camera_handler.release_cameras()
        
        # Close server client
        await self.server_client.close()
        
        logger.info("Edge device shutdown complete")


async def main():
    """Main application entry point."""
    # Configure logging
    logger.add(
        settings.log_file,
        rotation="1 day",
        retention="7 days",
        level=settings.log_level
    )
    
    logger.info("Starting SnitchSystem Edge Device")
    
    # Create and initialize edge device
    edge_device = EdgeDevice()
    
    if not await edge_device.initialize():
        logger.error("Failed to initialize edge device")
        return
    
    try:
        # Start detection loop
        await edge_device.run_detection_loop()
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        
    finally:
        await edge_device.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
