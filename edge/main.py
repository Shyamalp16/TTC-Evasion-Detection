"""
Main application entry point for the edge device.
"""
import asyncio
import cv2
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from config import settings

from camera_handler import CameraHandler
from detection_engine import DetectionEngine
from event_processor import EventProcessor
from server_client import ServerClient
from rules import ValidatorAndGateRules


class EdgeDevice:
    """Main edge device application."""
    
    def __init__(self):
        self.camera_handler = CameraHandler(settings.camera_urls)
        self.detection_engine = DetectionEngine()
        self.event_processor = EventProcessor()
        self.server_client = ServerClient()
        self.rules = ValidatorAndGateRules(settings)
        self.is_running = False
        self.frame_count = 0
        self._last_debug_log = 0.0
        self._last_detections: Dict[int, List[Dict[str, Any]]] = {}
        self._detection_task: Optional[asyncio.Task] = None
        self._frame_skip = max(1, settings.frame_skip)
        self._last_detection_trigger = 0.0
        
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
                    # Throttle warnings to avoid UI lag
                    now = time.time()
                    if now - self._last_debug_log > 2.0:
                        logger.warning("No frames received from cameras")
                        self._last_debug_log = now
                    await asyncio.sleep(0.1)
                    continue
                
                # Update preview as soon as we have frames
                self._update_preview(frames)

                # Schedule detection work based on frame skip and detection interval
                now = time.time()
                should_run_detection = (
                    self.frame_count % self._frame_skip == 0
                    and (now - self._last_detection_trigger) >= settings.detection_interval
                )

                if should_run_detection:
                    self._last_detection_trigger = now
                    if not self._detection_task or self._detection_task.done():
                        self._detection_task = asyncio.create_task(self._process_frames(list(frames)))

                self.frame_count += 1

                # Handle preview keypress (press 'q' to quit)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

                # Small delay primarily for UI pacing
                await asyncio.sleep(settings.preview_interval)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                await asyncio.sleep(1)

        if self._detection_task and not self._detection_task.done():
            try:
                await self._detection_task
            except Exception as e:
                logger.error(f"Pending detection task failed during shutdown: {e}")
    
    async def _process_frames(self, frames: List[Tuple[int, any]]):
        """Process frames for person detection."""
        try:
            # Run detection on all frames
            detection_results, crossed_person_ids = await self.detection_engine.detect_multiple_frames(frames)

            # Process each camera's detections
            frame_lookup = {camera_id: frame for camera_id, frame in frames}

            for camera_id, frame in frame_lookup.items():
                detections = detection_results.get(camera_id)
                if detections is not None:
                    self._last_detections[camera_id] = detections


                # Log tracking info
                if detections:
                    now = time.time()
                    if now - self._last_debug_log > 0.5:
                        logger.debug(
                            f"Camera {camera_id}: {len(detections)} tracked persons; "
                            + ", ".join(
                                f"ID:{det.get('person_id', '?')} {det.get('distance_m', -1.0):.1f}m"
                                for det in detections[:3]
                                if det.get('distance_m', -1.0) >= 0.0
                            )
                        )
                        self._last_debug_log = now

            # Update track state and run pose estimation for all detections
            for camera_id, detections in detection_results.items():
                for detection in detections:
                    person_id = detection.get('person_id')
                    if person_id is not None:
                        # Update track state with ROI information
                        validator_roi = self.detection_engine.validator_roi
                        if validator_roi:
                            self.rules.update_track(frame, detection, tuple(validator_roi))

                        # Run pose estimation once per track (with cooldown)
                        track_status = self.rules.get_status(person_id)
                        current_time = time.time()
                        last_pose_time = track_status.get("pose_timestamp", 0)
                        pose_cooldown = 0.5  # seconds between pose updates

                        # Only run pose estimation if we don't have recent pose data
                        if current_time - last_pose_time > pose_cooldown:
                            bbox = detection.get("bbox", [])
                            if len(bbox) >= 4:
                                try:
                                    # Crop the person from the frame
                                    x1, y1, x2, y2 = bbox
                                    cropped_person = frame[y1:y2, x1:x2]

                                    if cropped_person.size > 0:
                                        # Run pose estimation
                                        if self.detection_engine.pose_estimator is None:
                                            continue

                                        pose_result = self.detection_engine.pose_estimator.infer_pose(cropped_person)
                                        if pose_result:
                                            # Adjust keypoints back to full frame coordinates
                                            adjusted_keypoints = {}
                                            for name, kp in pose_result['keypoints'].items():
                                                adjusted_keypoints[name] = {
                                                    'x': kp['x'] + x1,
                                                    'y': kp['y'] + y1,
                                                    'z': kp['z'],
                                                    'visibility': kp['visibility']
                                                }

                                            # Store pose in track state
                                            track_status["pose_keypoints"] = adjusted_keypoints
                                            track_status["pose_timestamp"] = current_time

                                except Exception as e:
                                    pass  # Silently handle pose estimation failures

                        # Always use cached pose data for visualization if available
                        cached_pose = track_status.get("pose_keypoints")
                        if cached_pose:
                            detection["pose_keypoints"] = cached_pose

            # Handle gate crossings and create appropriate events
            if crossed_person_ids:
                logger.info(f"Gate crossings detected: {len(crossed_person_ids)} persons crossed")

                # Separate crossings by direction
                evasion_crossings = []
                normal_crossings = []

                # Check each crossed person to see their crossing direction
                for camera_id, detections in detection_results.items():
                    for detection in detections:
                        person_id = detection.get('person_id')
                        if person_id in crossed_person_ids:
                            # Find the person in the tracker to check crossing direction
                            if hasattr(self, 'detection_engine') and hasattr(self.detection_engine, 'person_tracker'):
                                tracker = self.detection_engine.person_tracker
                                if person_id in tracker.tracked_persons:
                                    person = tracker.tracked_persons[person_id]
                                    if person.crossing_direction == "right":
                                        evasion_crossings.append(detection)
                                    else:
                                        normal_crossings.append(detection)

                # Create detection events for normal crossings
                if normal_crossings:
                    await self.event_processor.add_detection_event(
                        camera_id=settings.primary_camera_id,
                        detections=normal_crossings
                    )

                # Create evasion events for left-to-right crossings
                if evasion_crossings:
                    logger.warning(f"Detected {len(evasion_crossings)} potential fare evasions (left-to-right crossings)")

                    import cv2

                    current_time = time.time()
                    camera_id = settings.primary_camera_id

                    # Capture snapshot from the primary camera frame for evasion events only
                    snapshot_data = None
                    primary_frame = frame_lookup.get(camera_id)
                    if primary_frame is not None:
                        # Encode frame as JPEG
                        success, encoded_image = cv2.imencode('.jpg', primary_frame)
                        if success:
                            snapshot_data = encoded_image.tobytes()
                            logger.debug(f"Captured evasion snapshot: {len(snapshot_data)} bytes")
                        else:
                            logger.warning("Failed to encode evasion snapshot")

                    # Send evasion event directly to server (following EventCreate schema)
                    simple_evasion_data = {
                        "gate_id": settings.gate_id,
                        "station_id": settings.station_id,
                        "event_type": "evasion",
                        "timestamp": current_time,  # Unix timestamp (server handles conversion)
                        "camera_id": camera_id,
                        "evasion_confidence": 1.0,
                        "num_detections": len(evasion_crossings),
                        "event_metadata": {
                            "detections": evasion_crossings,  # Put detections in metadata
                            "confidence_scores": [d.get("confidence", 0) for d in evasion_crossings],
                            "crossing_direction": "left_to_right",
                            "detection_method": "computer_vision",
                            "event_id": f"cv_evasion_{int(current_time)}"
                        }
                    }

                    # Send to server with snapshot for evasion events
                    asyncio.create_task(
                        self.server_client.send_event(simple_evasion_data, snapshot_data)
                    )

                    # Also send detection events
                    await self.event_processor.add_detection_event(
                        camera_id=camera_id,
                        detections=evasion_crossings
                    )

            # Periodically cleanup old tracks from rules engine
            self.rules.cleanup_old_tracks()

            # Check for batched detections to send
            await self._send_batched_detections()
            
        except Exception as e:
            logger.error(f"Error processing frames: {e}")

    def _update_preview(self, frames: List[Tuple[int, any]]):
        """Render preview windows without blocking detection loop."""
        for camera_id, frame in frames:
            detections = self._last_detections.get(camera_id) or []
            try:
                # Always draw detections (which includes ROIs) even with empty detections list
                annotated = self.detection_engine.draw_detections(frame, detections)
                cv2.imshow(f"Camera {camera_id}", annotated)
            except Exception:
                pass
    
    async def _send_batched_detections(self):
        """Send batched detection events."""
        try:
            batch = await self.event_processor.get_pending_detections_batch()
            if batch:
                # Send batch as single request
                asyncio.create_task(
                    self.server_client.send_detection_batch(batch)
                )
        except Exception as e:
            logger.error(f"Error sending batched detections: {e}")
    
    
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
        
        # Close preview windows if any
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        logger.info("Edge device shutdown complete")


async def main():
    """Main application entry point."""
    # Configure logging
    logger.remove()  # remove default console sink
    # File sink
    logger.add(
        settings.log_file,
        rotation="1 day",
        retention="7 days",
        level=settings.file_log_level
    )
    # Optional console sink with higher level to reduce overhead
    if settings.enable_console_logs:
        logger.add(lambda msg: print(msg, end=""), level=settings.console_log_level)
    
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
