from typing import Tuple
import numpy as np
from shapely.geometry import box as _box
from shapely import affinity as _aff

def oriented_box_polygon(cx: float, cy: float, length_m: float, width_m: float, yaw_rad: float):
    """Return a Shapely polygon for a rectangle centered at (cx,cy), yaw in radians."""
    # axis-aligned box centered at origin
    poly = _box(-length_m/2.0, -width_m/2.0, length_m/2.0, width_m/2.0)
    # rotate about origin (degrees for shapely), then translate
    poly = _aff.rotate(poly, yaw_rad * 180.0/np.pi, origin=(0.0, 0.0), use_radians=False)
    poly = _aff.translate(poly, cx, cy)
    return poly

def box_distance(poly_a, poly_b) -> float:
    """Geometric distance (meters) between two polygons (0 if they intersect)."""
    # shapely distance is exact for polygons
    return float(poly_a.distance(poly_b))
