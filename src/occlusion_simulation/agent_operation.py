import numpy as np
import skgeom as sg
from typing import List, Tuple
import src.occlusion_simulation.polygon_generation as poly_gen
from src.data.sdd_dataloader import StanfordDroneAgent


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
