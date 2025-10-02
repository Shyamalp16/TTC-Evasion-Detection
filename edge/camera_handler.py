"""
Camera handling module for RTSP streams.
"""
import os
import cv2
import asyncio
from typing import Optional, Tuple, Union
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
            logger.info(f"Initializing cameras with sources: {self.camera_urls}")
            for i, url in enumerate(self.camera_urls):
                cap = None
                tried = []
                if isinstance(url, str) and url.isdigit():
                    device_index = int(url)
                    if os.name == "nt":
                        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW); tried.append("CAP_DSHOW")
                        if not cap or not cap.isOpened():
                            cap.release() if cap else None
                            cap = cv2.VideoCapture(device_index, cv2.CAP_MSMF); tried.append("CAP_MSMF")
                    else:
                        cap = cv2.VideoCapture(device_index); tried.append("DEFAULT")
                else:
                    ffmpeg_backend_flag = getattr(cv2, "CAP_FFMPEG", None)
                    rtsp_url = url  
                    try:
                        if settings.ffmpeg_capture_options:
                            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = settings.ffmpeg_capture_options
                    except Exception:
                        pass
                    if ffmpeg_backend_flag is not None:
                        cap = cv2.VideoCapture(rtsp_url, ffmpeg_backend_flag); tried.append("FFMPEG")
                    if not cap or not cap.isOpened():
                        cap.release() if cap else None
                        cap = cv2.VideoCapture(rtsp_url); tried.append("SOURCE")
                
                # Configure for low-latency streaming
                if cap:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 60)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                # Fallback to default webcam 0 if open failed and this looks like a stream/path
                if (not cap or not cap.isOpened()) and not (isinstance(url, int) or (isinstance(url, str) and url.isdigit())):
                    if settings.disable_webcam_fallback:
                        logger.error(f"Primary source failed for camera {i}: {url}. Webcam fallback disabled.")
                    else:
                        logger.warning(f"Primary source failed for camera {i}: {url}. Fallback to webcam 0.")
                        if os.name == "nt":
                            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); tried.append("FALLBACK_DSHOW_0")
                            if not cap or not cap.isOpened():
                                cap.release() if cap else None
                                cap = cv2.VideoCapture(0, cv2.CAP_MSMF); tried.append("FALLBACK_MSMF_0")
                        else:
                            cap = cv2.VideoCapture(0); tried.append("FALLBACK_DEFAULT_0")
                
                if not cap or not cap.isOpened():
                    logger.error(f"Failed to open camera {i} using attempts: {tried}. Source: {url}")
                    return False
                    
                self.cameras.append(cap)
                logger.info(f"Camera {i} initialized (attempts: {tried}): {url}")
                
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
        
        if not ret or frame is None:
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
            try:
                cap.release()
            except Exception:
                pass
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
