import numpy as np
import skgeom as sg
import shapely.geometry as sp
from typing import List


def shapely_poly_2_skgeom_poly(poly: sp.Polygon) -> sg.Polygon:
    return sg.Polygon([sg.Point2(*coord) for coord in poly.exterior.coords[:-1]][::-1])


def skgeom_poly_2_shapely_poly(poly: sg.PolygonWithHoles) -> sp.Polygon:
    return sp.Polygon(shell=poly.outer_boundary().coords, holes=[hole.coords for hole in poly.holes])


def trajectory_2_segment2_list(traj: np.array) -> List[sg.Segment2]:
    segments = []
    for idx in range(traj.shape[0] - 1):
        start = sg.Point2(traj[idx, 0], traj[idx, 1])
        end = sg.Point2(traj[idx + 1, 0], traj[idx + 1, 1])
        segments.append(sg.Segment2(start, end))
    return segments
