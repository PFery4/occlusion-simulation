import numpy as np
from scipy.interpolate import interp1d
import skgeom as sg
import shapely.geometry as sp
import functools
from typing import List, Tuple, Union
import src.occlusion_simulation.polygon_generation as poly_gen
import src.occlusion_simulation.type_conversion as type_conv
import src.occlusion_simulation.visibility as visibility
from src.data.sdd_agent import StanfordDroneAgent


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
        past_window: np.array,
        min_travl_dist: float
) -> Tuple[np.array, np.array]:
    """
    returns a list of candidate agents to be the targets for occlusion simulation, with corresponding sampling
    probabilities. target agents must be:
    - in movement
    - fully observed over the entire full_window
    the sampling probabilities are proportional to the total distance travelled by each agent in their past.
    """

    # checking that the target agents are fully observed over the entire window
    # (in practice: [-T_obs:T_pred+1], INCLUDING the +1)
    fully_observed = np.array([agent.get_traj_section(full_window).shape[0] == full_window.shape[0]
                               for agent in agent_list])

    # checking that the target agents are moving
    distances = np.array([np.linalg.norm(pasttraj[-1] - pasttraj[0]) for pasttraj in
                          [agent.get_traj_section(past_window) for agent in agent_list]])
    not_idle = (distances > 1e-8)

    # checking that the agents meet the distance requirement: having travelled min_travl_dist over the past window
    # (this is to help prevent the simulation of occlusions over agents who are standing still and/or barely moving)
    travl_dists = np.array([agent.get_travelled_distance(past_window) for agent in agent_list])
    moving = (travl_dists > min_travl_dist)

    candidates = functools.reduce(
        lambda arr_a, arr_b: np.logical_and(arr_a, arr_b),
        (fully_observed, not_idle, moving)
    )
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
        min_obs: int, min_reobs: int, min_occl:int, tol: float = 1e-8
) -> Tuple[int, int]:

    full_window = np.concatenate([past_window, future_window])
    full_traj = agent.get_traj_section(full_window)
    moving = (np.linalg.norm(full_traj[1:] - full_traj[:-1], axis=1) >= tol)

    past_mask = np.in1d(full_window, past_window)
    past_mask[:min_obs-1] = False
    past_occl_indices = np.nonzero(np.logical_and(past_mask[:past_window.shape[0]], moving[:past_window.shape[0]]))[0]
    t_occl = int(np.random.choice(past_occl_indices))

    future_mask = np.in1d(full_window, future_window)
    if min_reobs != 0:
        future_mask[-min_reobs:] = False
    future_mask[t_occl+1:t_occl+1+min_occl] = False
    future_occl_indices = np.nonzero(np.logical_and(
        future_mask[-future_window.shape[0]:], moving[-future_window.shape[0]:]
    ))[0]

    t_disoccl = int(np.random.choice(future_occl_indices) + past_window.shape[0])
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
