import numpy as np
from scipy.interpolate import interp1d
import skgeom as sg
import shapely.geometry as sp
import functools
from typing import List, Tuple, Union
import src.occlusion_simulation.polygon_generation as poly_gen
import src.occlusion_simulation.type_conversion as type_conv
import src.occlusion_simulation.visibility as visibility
from src.data.sdd_dataloader import StanfordDroneAgent


def interpolate_trajectory(traj: np.array, dt: float = 1.0) -> np.array:
    x = traj[:, 0]
    y = traj[:, 1]

    dist = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    dist = np.concatenate([[0], dist])

    interp_x = interp1d(dist, x)
    interp_y = interp1d(dist, y)

    new_dist = np.arange(0, dist[-1], dt)

    new_x = interp_x(new_dist)
    new_y = interp_y(new_dist)

    return np.column_stack((new_x, new_y))


def target_agent_candidate_indices(
        agent_list: List[StanfordDroneAgent],
        full_window: np.array,
        past_window: np.array
) -> Tuple[np.array, np.array]:
    """
    returns a list of candidate agents to be the targets for occlusion simulation, with corresponding sampling
    probabilities. target agents must be:
    - in movement
    - fully observed over the entire full_window
    """

    # checking that the target agents are fully observed over the entire window
    # (in practice: [-T_obs:T_pred+1], INCLUDING the +1)
    fully_observed = np.array([agent.get_traj_section(full_window).shape[0] == full_window.shape[0]
                               for agent in agent_list])

    # checking that the target agents are moving
    distances = np.array([np.linalg.norm(pasttraj[-1] - pasttraj[0]) for pasttraj in
                          [agent.get_traj_section(past_window) for agent in agent_list]])
    moving = (distances > 1e-8)
    candidates = np.logical_and(fully_observed, moving)

    # todo: add check that all subsequent positions are different

    return np.asarray(candidates).nonzero()[0], distances[candidates]/sum(distances[candidates])


def select_random_target_agents(
        candidate_indices: np.array,
        sampling_probabilities: np.array,
        n: int = 1
) -> np.array:
    """
    selects a random subset of agents present within the scene.
    n agents will be sampled (possibly fewer if there aren't enough agents).
    the output provides the indices in agent_list pointing at our target agents.
    """
    if n > len(candidate_indices):
        # print(f"returning all available candidates: only {sum(is_moving)} moving agents in the scene")
        return candidate_indices
    return np.asarray(np.random.choice(candidate_indices, n, replace=False, p=sampling_probabilities))


def target_agent_no_ego_wedges(boundary: sg.Polygon, traj: np.array, offset: float, angle: float) -> List[sg.Polygon]:
    # offset: distance to pull the wedges "inward"
    u_traj = np.array(traj[-1] - traj[0])
    u_traj /= np.linalg.norm(u_traj)
    wedge_1 = poly_gen.bounded_wedge(
        p=(np.array(traj[0])) + u_traj * offset,
        u=u_traj,
        theta=float(np.radians(angle)),
        boundary=boundary
    )
    wedge_2 = poly_gen.bounded_wedge(
        p=(np.array(traj[-1])) - u_traj * offset,
        u=-u_traj,
        theta=float(np.radians(angle)),
        boundary=boundary
    )
    return [wedge_1, wedge_2]


def generate_occlusion_timesteps(
        agent: StanfordDroneAgent, past_window: np.array, future_window: np.array,
        min_obs: int, min_reobs: int, tol: float = 1e-8
) -> Tuple[int, int]:

    full_window = np.concatenate([past_window, future_window])
    full_traj = agent.get_traj_section(full_window)
    displacements = np.linalg.norm(full_traj[1:] - full_traj[:-1], axis=1)
    moving = (displacements >= tol)
    past_mask = np.in1d(full_window, past_window)
    past_mask[:min_obs-1] = False
    past_occl_indices = np.nonzero(np.logical_and(past_mask[:past_window.shape[0]], moving[:past_window.shape[0]]))[0]
    t_occl = int(np.random.choice(past_occl_indices))

    future_mask = np.in1d(full_window, future_window)
    if min_reobs != 0:
        future_mask[-min_reobs:] = False
    future_occl_indices = np.nonzero(np.logical_and(
        future_mask[-future_window.shape[0]:], moving[-future_window.shape[0]:]
    ))[0]

    t_disoccl = int(np.random.choice(future_occl_indices) + past_window.shape[0])

    # for agent in range(n_agents):
    #     t_occl = np.random.randint(min_obs - 1, T_obs - 1)
    #     t_disoccl = np.random.randint(T_pred - min_reobs + 1) + T_obs
    #     occlusion_windows.append((t_occl, t_disoccl))
    return t_occl, t_disoccl


def trajectory_buffers(
        agent_list: List[StanfordDroneAgent],
        time_window: Union[None, np.array],
        buffer_radius: float
) -> List[sg.Polygon]:
    if time_window is not None:
        return [
            type_conv.shapely_poly_2_skgeom_poly(
                sp.LineString(agent.get_traj_section(time_window)).buffer(buffer_radius)
            )
            for agent in agent_list
        ]
    return [
        type_conv.shapely_poly_2_skgeom_poly(sp.LineString(agent.fulltraj).buffer(buffer_radius))
        for agent in agent_list
    ]


def instantaneous_visibility_polygons(
        agents: List[StanfordDroneAgent],
        target_agent_indices: List[int],
        agent_radius: float,
        interp_dt: float,
        time_window: np.array,
        boundary: sg.Polygon
) -> List[sg.PolygonSet]:
    # TODO: ONLY AS AN OPTIONAL IMPROVEMENT IF WE HAVE NOTHING TO DO AT SOME POINT.
    # TODO: Maybe improve for better runtime (by linetracing for t=-T_obs and t=0, then buffering with -d_agent using
    # TODO: shapely. This prevents the necessity to compute the visibility polygon at every timestep.
    # TODO: WIP WIP WIP, FRAGMENTED VISIPOLYGONS (reason unknown)
    target_visipolys = []

    for idx in target_agent_indices:
        other_agents = agents.copy()
        other_agents.pop(idx)

        target_traj = agents[idx].get_traj_section(time_window)

        other_trajs = np.array([agent.get_traj_section(time_window) for agent in other_agents])
        other_vecs = other_trajs - target_traj
        other_us = other_vecs / np.sqrt(np.einsum('...i,...i', other_vecs, other_vecs))[..., np.newaxis]
        out_points = other_trajs + 3000 * other_us

        # print(target_traj.shape)
        # print(other_trajs.shape)
        # print(other_vecs.shape)
        # print(other_vecs[0])
        # print(other_us[0])
        # print(other_us.shape)
        # print(out_points[0])
        shifted_points = np.roll(other_trajs, -1, axis=1)
        other_segments = np.stack([other_trajs, shifted_points], axis=-1).transpose((0, 1, 3, 2))
        print(other_trajs[0][0])
        print(shifted_points[0][0])

        out_shifted = np.roll(out_points, -1, axis=1)
        out_segments = np.stack([out_points, out_shifted], axis=-1).transpose((0, 1, 3, 2))
        print(out_points[0][0])
        print(out_shifted[0][0])

        poly_block = np.stack([other_trajs, shifted_points, out_shifted, out_points], axis=-1).transpose((0, 1, 3, 2))
        print(poly_block[0][0])
        print(poly_block.shape)
        poly_block = poly_block[:, :-1, :, :]
        print(poly_block.shape)

        occl_polys = []
        for other_agent in poly_block:
            # polygons = [sg.Polygon(np.unique(poly_line, axis=0)) for poly_line in other_agent]
            polygons = [type_conv.shapely_poly_2_skgeom_poly(sp.Polygon(np.unique(poly_line, axis=0)).buffer(agent_radius))
                        for poly_line in other_agent]

            # print(polygons[0])
            # print(polygons_sp[0])
            # print(zblu)
            # [poly.reverse_orientation() for poly in polygons if poly.orientation() == sg.Sign.CLOCKWISE]
            # [print(poly.orientation()) for poly in polygons]
            occl_poly = functools.reduce(
                lambda poly_a, poly_b: sg.boolean_set.join(poly_a, poly_b),
                polygons
            )
            # print(occl_poly)
            # print(type(occl_poly))
            occl_polys.append(occl_poly)
        occl_polys = sg.PolygonSet(occl_polys)
        visi_poly = sg.PolygonSet(boundary).difference(occl_polys)
        # print(visi_poly)
        # print(type(visi_poly))
        # print(len(visi_poly.polygons))
        # print(visi_poly.polygons)
        # print(zblu)
        target_visipolys.append(visi_poly)

    # print(target_visipolys)
    # print(zblu)
    raise NotImplementedError       # return target_visipolys


def trajectory_visibility_polygons(
        agents: List[StanfordDroneAgent],
        target_agent_indices: List[int],
        agent_visipoly_buffers: List[sg.Polygon],
        time_window: np.array,
        boundary: sg.Polygon
) -> List[sg.PolygonSet]:
    """
    For each target agent, compute the regions in space from which their full trajectory can be observed without
    obstruction from other agent's trajectories. This is performed by intersecting the visibility polygons for each of
    that agent's trajectory coordinates, with the occluders being the remaining agents.

    :param agents: the full list of agents present in the scene.
    :param target_agent_indices: the indices within agents to consider as target agents (and for which to generate a
    trajectory visibility polygon)
    :param agent_visipoly_buffers: polygon representation of agents' trajectories (used as occluders)
    :param time_window: time window to consider for the target agent
    :param boundary: exterior boundary, necessary to limit the visibility polygon.
    :return: a list containing the trajectory visibility polygons for each of the target agents
    """
    trajectory_visipolys = []
    for idx in target_agent_indices:
        traj = agents[idx].get_traj_section(time_window)

        # to generate the regions within which every coordinate of the target agent is visible, we first need
        # the buffers of every *other* agent
        other_buffers = agent_visipoly_buffers.copy()
        other_buffers.pop(idx)

        # creating the sg.arrangement.Arrangement object necessary to compute the visibility polygons
        scene_segments = list(boundary.edges)
        [scene_segments.extend(poly.edges) for poly in other_buffers]
        scene_arr = sg.arrangement.Arrangement()
        [scene_arr.insert(seg) for seg in scene_segments]

        # generating visibility polygons along every position of the target agent's trajectory,
        # and computing the intersection of all those visibility polygons: this corresponds to the regions in
        # the scene from which every coordinate of the target agent can be seen.
        traj_fully_observable = functools.reduce(
            lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
            [sg.PolygonSet(visibility.visibility_polygon(point, arrangement=scene_arr))
             for point in interpolate_trajectory(traj, dt=10)]
        )
        trajectory_visipolys.append(traj_fully_observable)

    return trajectory_visipolys


def verify_target_agents_occlusion_pattern(
        visibility_polygon: sg.Polygon,
        full_window: np.array,
        agents: List[StanfordDroneAgent],
        target_agent_indices: List[int],
        occlusion_windows: List[Tuple[int, int]],
):
    """
    for each target agent, verify that the agent follows the occlusion pattern visible -> occluded -> visible, according
    to the selected occlusion windows.
    """
    patterns_correct = []
    for target_idx, occl_window in zip(target_agent_indices, occlusion_windows):
        visi_pattern_expected = np.ones_like(full_window)
        visi_pattern_expected[occl_window[0]+1:occl_window[1]] = 0
        visi_pattern_expected = visi_pattern_expected.astype(bool)

        agent = agents[target_idx]
        fulltraj = agent.get_traj_section(full_window)
        visi_pattern_sim = np.array(
            [visibility_polygon.oriented_side(sg.Point2(*point)) == sg.Sign.POSITIVE
             for point in fulltraj], dtype=bool
        )

        pattern_correct = all(visi_pattern_sim == visi_pattern_expected)
        patterns_correct.append(pattern_correct)
    return patterns_correct
