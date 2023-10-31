import torch
Tensor = torch.Tensor
import numpy as np
import skgeom as sg
from typing import List, Tuple
from src.data.sdd_agent import StanfordDroneAgent


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


def torch_compute_visipoly(
        ego_point: Tensor,
        occluder: Tensor,
        boundary: sg.Polygon
) -> sg.Polygon:
    # ego_point [2]
    # occluder [2, 2]
    # boundary [2]
    ego_visi_arrangement = sg.arrangement.Arrangement()
    ego_visi_arrangement.insert(sg.Segment2(sg.Point2(*occluder[0]), sg.Point2(*occluder[1])))
    [ego_visi_arrangement.insert(segment) for segment in list(boundary.edges)]
    return visibility_polygon(ego_point=tuple(ego_point), arrangement=ego_visi_arrangement)


def agent_occlusion_masks(
        agents: List[StanfordDroneAgent],
        time_window: np.array,
        ego_visipoly: sg.Polygon,
) -> np.array:
    agent_masks = []
    for agent in agents:
        agent_masks.append(occlusion_mask(points=agent.get_traj_section(time_window), ego_visipoly=ego_visipoly))
    return np.stack(agent_masks)        # [N, T]


def occlusion_mask(
        points: np.array,
        ego_visipoly: sg.Polygon
) -> np.array:
    # points.shape = [N, 2]
    return np.array([
        ego_visipoly.oriented_side(sg.Point2(*point)) == sg.Sign.POSITIVE
        for point in points
    ])


def torch_occlusion_mask(
        points: Tensor,
        ego_visipoly: sg.Polygon
) -> Tensor:
    # points [N, 2]
    return torch.Tensor([
        ego_visipoly.oriented_side(sg.Point2(*point)) == sg.Sign.POSITIVE for point in points
    ])
