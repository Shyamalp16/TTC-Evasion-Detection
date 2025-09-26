"""
Camera handling module for RTSP streams.
"""
import cv2
import asyncio
from typing import Optional, Tuple
from loguru import logger
from config import settings


class CameraHandler:
    """Handles RTSP camera streams for dual camera setup."""
    
    def __init__(self, camera_urls: list):
        self.camera_urls = camera_urls
        self.cameras = []
        self.is_running = False
        
    async def initialize_cameras(self) -> bool:
        """Initialize camera connections."""
        try:
            for i, url in enumerate(self.camera_urls):
                cap = cv2.VideoCapture(url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    logger.error(f"Failed to open camera {i}: {url}")
                    return False
                    
                self.cameras.append(cap)
                logger.info(f"Camera {i} initialized: {url}")
                
            return True
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return False
    
    async def read_frame(self, camera_index: int) -> Optional[Tuple[int, any]]:
        """Read frame from specified camera."""
        if camera_index >= len(self.cameras):
            return None
            
        cap = self.cameras[camera_index]
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Failed to read frame from camera {camera_index}")
            return None
            
        return camera_index, frame
    
    async def read_all_frames(self) -> list:
        """Read frames from all cameras."""
        frames = []
        tasks = []
        
        for i in range(len(self.cameras)):
            task = self.read_frame(i)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, tuple) and result[1] is not None:
                frames.append(result)
                
        return frames
    
    def release_cameras(self):
        """Release all camera resources."""
        for i, cap in enumerate(self.cameras):
            cap.release()
            logger.info(f"Camera {i} released")
        self.cameras.clear()
    
    async def health_check(self) -> dict:
        """Check camera health status."""
        status = {
            "total_cameras": len(self.cameras),
            "active_cameras": 0,
            "camera_status": []
        }
        
        for i, cap in enumerate(self.cameras):
            is_opened = cap.isOpened()
            status["camera_status"].append({
                "camera_id": i,
                "url": self.camera_urls[i],
                "is_connected": is_opened
            })
            if is_opened:
                status["active_cameras"] += 1
                
        return status
