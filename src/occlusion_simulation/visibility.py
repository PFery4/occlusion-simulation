import numpy as np
import skgeom as sg
from typing import List, Tuple


def visibility_polygon(ego_point: Tuple[float, float], arrangement: sg.arrangement.Arrangement) -> sg.Polygon:
    visibility = sg.RotationalSweepVisibility(arrangement)
    # visibility = sg.TriangularExpansionVisibility(arrangement)
    origin = sg.Point2(*ego_point)
    face = arrangement.find(origin)
    vx = visibility.compute_visibility(origin, face)
    # convert Arrangement to sg.Polygon
    return sg.Polygon([pt.point() for pt in vx.vertices])


def compute_visibility_polygon(
        ego_point: Tuple[float, float],
        occluders: List[Tuple[np.array, np.array]],
        boundary: sg.Polygon
) -> sg.Polygon:
    ego_visi_arrangement = sg.arrangement.Arrangement()
    [ego_visi_arrangement.insert(sg.Segment2(sg.Point2(*occluder_coords[0]), sg.Point2(*occluder_coords[1])))
     for occluder_coords in occluders]
    [ego_visi_arrangement.insert(segment) for segment in list(boundary.edges)]

    return visibility_polygon(ego_point=ego_point, arrangement=ego_visi_arrangement)
