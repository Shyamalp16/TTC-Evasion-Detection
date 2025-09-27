"""
ROI (Region of Interest) helper functions for SnitchSystem.

Provides utilities for:
- Checking if points are within bounding box ROIs
- Detecting when objects cross gate lines for fare evasion detection
"""
from typing import Tuple, List, Dict, Any, Optional
import math


def point_in_roi(point: Tuple[int, int], roi: List[int]) -> bool:
    """
    Check if a point is within a bounding box ROI.

    Args:
        point: (x, y) coordinates of the point
        roi: [x1, y1, x2, y2] bounding box coordinates where
             (x1,y1) is top-left, (x2,y2) is bottom-right

    Returns:
        True if point is within the ROI, False otherwise
    """
    x, y = point
    x1, y1, x2, y2 = roi

    return x1 <= x <= x2 and y1 <= y <= y2


def _point_to_line_distance(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
    """
    Calculate the perpendicular distance from a point to a line segment.

    Args:
        point: (x, y) coordinates of the point
        line_start: (x, y) start point of the line
        line_end: (x, y) end point of the line

    Returns:
        Distance from point to line segment
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Vector from line_start to line_end
    dx = x2 - x1
    dy = y2 - y1

    # If line has zero length, return distance to start point
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)

    # Parameter t represents position along line segment (0 = start, 1 = end)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx**2 + dy**2)))

    # Closest point on line segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # Distance to closest point
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def _get_line_side(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> int:
    """
    Determine which side of a line a point is on.

    Args:
        point: (x, y) coordinates of the point
        line_start: (x, y) start point of the line
        line_end: (x, y) end point of the line

    Returns:
        1 if point is to the right of the line (when traversing from start to end)
        -1 if point is to the left of the line
        0 if point is on the line
    """
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Cross product to determine side
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    if cross > 0:
        return 1  # right side
    elif cross < 0:
        return -1  # left side
    else:
        return 0  # on line


def crosses_line(bbox_center: Tuple[float, float], gate_line: List[int], previous_center: Optional[Tuple[float, float]] = None) -> bool:
    """
    Check if a bounding box center has crossed or is crossing a gate line.

    This function supports both simple position-based crossing (if previous_center is None)
    and motion-based crossing detection (if previous_center is provided).

    Args:
        bbox_center: (x, y) center coordinates of the bounding box
        gate_line: [x1, y1, x2, y2] line coordinates
        previous_center: Optional (x, y) previous center position for motion detection

    Returns:
        True if the bbox center crosses the line, False otherwise
    """
    cx, cy = bbox_center
    x1, y1, x2, y2 = gate_line

    # For vertical lines (x1 == x2), check x-coordinate crossing
    if x1 == x2:
        line_x = x1
        if previous_center is not None:
            prev_x, prev_y = previous_center
            # Check if movement crosses the vertical line
            return (prev_x < line_x and cx >= line_x) or (prev_x > line_x and cx <= line_x)
        else:
            # Simple check if center is at or beyond the line
            return cx >= line_x

    # For horizontal lines (y1 == y2), check y-coordinate crossing
    elif y1 == y2:
        line_y = y1
        if previous_center is not None:
            prev_x, prev_y = previous_center
            # Check if movement crosses the horizontal line
            return (prev_y < line_y and cy >= line_y) or (prev_y > line_y and cy <= line_y)
        else:
            # Simple check if center is at or beyond the line
            return cy >= line_y

    # For diagonal lines, use the more complex crossing detection
    else:
        if previous_center is None:
            # Without previous position, just check if point is close to the line
            distance = _point_to_line_distance(bbox_center, (x1, y1), (x2, y2))
            return distance <= 5.0  # 5 pixel threshold
        else:
            # Use motion-based crossing detection for diagonal lines
            line_start = (x1, y1)
            line_end = (x2, y2)

            current_side = _get_line_side(bbox_center, line_start, line_end)
            previous_side = _get_line_side(previous_center, line_start, line_end)

            # If sides are different, we have a crossing
            return current_side != 0 and previous_side != 0 and current_side != previous_side
