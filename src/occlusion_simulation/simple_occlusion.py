import os.path
import argparse
import numpy as np
import shapely.geometry as sp
import skgeom as sg
import functools
import itertools
from typing import List, Tuple
import src.occlusion_simulation.polygon_generation as poly_gen
import src.occlusion_simulation.type_conversion as type_conv
import src.occlusion_simulation.polygon_operation as poly_op
import src.occlusion_simulation.agent_operation as agent_op
import src.occlusion_simulation.visibility as visibility
import src.data.config as conf
from src.data.sdd_dataloader import StanfordDroneDataset
from src.data.sdd_agent import StanfordDroneAgent


def simulate_occlusions(
        config: dict,
        image_res: Tuple[int, int],
        agents: List[StanfordDroneAgent],
        past_window: np.array,
        future_window: np.array
):
    min_travl_dist = config["min_travl_dist"]

    min_obs = config["min_obs"]
    min_reobs = config["min_reobs"]
    min_occl = config["min_occl"]

    d_border = config["d_border"]
    d_min_occl_ag = config["d_min_occl_ag"]
    d_min_occl_ego = config["d_min_occl_ego"]
    k_ag_ego_distance = config["k_ag_ego_distance"]
    d_min_ag_ego = (d_min_occl_ego + d_min_occl_ag) * k_ag_ego_distance

    taper_angle = config["target_angle"]
    r_agents = config["r_agents"]

    simulation_dict = {
        "target_agent_indices": None,
        "occlusion_windows": None,
        "occlusion_target_coords": None,
        "frame_box": None,
        "agent_visipoly_buffers": None,
        "no_occluder_buffers": None,
        "no_ego_buffers": None,
        "no_ego_wedges": None,
        "targets_fullobs_regions": None,
        "yes_ego_triangles": None,
        "ego_point": None,
        "ego_buffer": None,
        "p1_area": None,
        "p1_triangles": None,
        "p1_visipolys": None,
        "p2_area": None,
        "p2_triangles": None,
        "occluders": None,
        "occluded_regions": None
    }

    # full window contains the past and future timesteps, with an extra timestep after the prediction horizon.
    # the extra timestep is used to help define the disocclusion point of the target agent, which might happen one
    # timestep after the full prediction horizon (ie, the target agent makes a reappearance from occlusion only
    # after T_pred timesteps).
    future_window_plus_one = np.concatenate((future_window, [2 * future_window[-1] - future_window[-2]]))
    full_window = np.concatenate((past_window, future_window_plus_one))

    # set safety perimeter around the edges of the scene
    d_border_px = int((d_border/100 * np.linalg.norm(image_res)) // 10 * 11)
    scene_boundary = poly_gen.default_rectangle(image_res)
    frame_box = poly_op.skgeom_extruded_polygon(scene_boundary, d_border=d_border_px)
    simulation_dict["frame_box"] = frame_box

    # define agent_buffers, a list of sg.Polygons
    # corresponding to the past trajectories of every agent, inflated by some small radius (used for computation of
    # visibility polygons, in order to place the ego_point)
    agent_visipoly_buffers = agent_op.trajectory_buffers(agents, past_window, r_agents)
    simulation_dict["agent_visipoly_buffers"] = agent_visipoly_buffers

    # define no_ego_buffers, a list of sg.Polygons, within which we wish not to place the ego
    no_ego_buffers = agent_op.trajectory_buffers(agents, full_window[:-1], d_min_ag_ego)
    no_ego_buffers = sg.PolygonSet(no_ego_buffers)
    simulation_dict["no_ego_buffers"] = no_ego_buffers

    # define no_occluder_zones, a list of sg.Polygons, within which we wish not to place any virtual occluder
    no_occluder_buffers = agent_op.trajectory_buffers(agents, full_window[:-1], d_min_occl_ag)
    simulation_dict["no_occluder_buffers"] = no_occluder_buffers

    # BEGINNING OF CHECK FOR VALIDITY OF EGO PLACEMENT ##############################################################
    # choose agents within the scene whose trajectory we would like to occlude virtually
    target_agent_candidates, target_probabilities = agent_op.target_agent_candidate_indices(
        agent_list=agents, full_window=full_window, past_window=past_window, min_travl_dist=min_travl_dist
    )
    if np.size(target_agent_candidates) == 0:
        raise ValueError("No valid candidates available for occlusion simulation.")

    target_agent_indices = None
    occlusion_windows = None
    no_ego_wedges = None
    targets_fullobs_regions = None
    yes_ego_triangles = []

    while np.size(target_agent_candidates) != 0 and len(yes_ego_triangles) == 0:
        target_agent_indices = agent_op.select_random_target_agents(
            target_agent_candidates, target_probabilities, n=1
        )

        # generate occlusion windows -> List[Tuple[int, int]]
        # each item provides two timesteps for each target agent:
        # - the first one corresponds to the last observed timestep before occlusion
        # - the second one corresponds to the first re-observed timestep before reappearance
        occlusion_windows = [agent_op.generate_occlusion_timesteps(
            agent=agents[idx],
            past_window=past_window,
            future_window=future_window_plus_one,
            min_obs=min_obs, min_reobs=min_reobs, min_occl=min_occl
        ) for idx in target_agent_indices]

        # define no_ego_wedges, a sg.PolygonSet, containing sg.Polygons within which we wish not to place the ego
        # the wedges are placed at the extremeties of the target agents, in order to prevent ego placements directly aligned
        # with the target agents' trajectories
        no_ego_wedges = sg.PolygonSet(list(itertools.chain(*[
            agent_op.target_agent_no_ego_wedges(
                scene_boundary, agents[idx].get_traj_section(
                    full_window[occlusion_windows[i][0]:occlusion_windows[i][1]]
                ), d_min_ag_ego, taper_angle
            )
            for i, idx in enumerate(target_agent_indices)
        ])))

        # list: a given item is a sg.PolygonSet, which describes the regions in space from which
        # every timestep of that agent can be directly observed, unobstructed by other agents
        # (specifically, by their agent_buffer)
        targets_fullobs_regions = agent_op.trajectory_visibility_polygons(
            agents=agents,
            target_agent_indices=target_agent_indices,
            agent_visipoly_buffers=agent_visipoly_buffers,
            time_window=full_window[:-1],
            boundary=scene_boundary
        )

        # reducing into a single sg.PolygonSet
        targets_fullobs_regions = functools.reduce(
            lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
            targets_fullobs_regions
        )

        # the regions within which we sample our ego are those within which target agents' full trajectories are
        # observable, minus the boundaries and no_ego_zones we set previously.
        # we will need to triangulate those regions in order to sample a point
        # this can't be done in scikit-geometry (maybe it can?), so we're doing it with shapely instead
        # (see inside triangulate_polyset function)
        yes_ego_triangles = poly_op.triangulate_polyset(
            targets_fullobs_regions.difference(
                no_ego_buffers.union(no_ego_wedges).union(frame_box)
            )
        )

        # Here, verify that we do have triangles, otherwise remove target agent candidate, and do it again
        if len(yes_ego_triangles) == 0:
            # remove a random target_agent candidate
            removed_target_idx = np.random.choice(target_agent_indices)
            keep_index = (target_agent_candidates != removed_target_idx)
            target_agent_candidates = target_agent_candidates[keep_index]
            target_probabilities = target_probabilities[keep_index]
            target_probabilities /= sum(target_probabilities)

    if len(yes_ego_triangles) == 0:
        raise ValueError(f"No placement of ego possible: yes_ego_triangle->{yes_ego_triangles}")

    simulation_dict["target_agent_indices"] = target_agent_indices
    simulation_dict["occlusion_windows"] = occlusion_windows
    simulation_dict["no_ego_wedges"] = no_ego_wedges
    simulation_dict["targets_fullobs_regions"] = targets_fullobs_regions
    simulation_dict["yes_ego_triangles"] = yes_ego_triangles

    # to those occlusion window timesteps, we compute the corresponding occlusion coordinates for each target agent
    occlusion_target_coords = [(agents[idx].position_at_timestep(full_window[occlusion_window[0]]),
                                agents[idx].position_at_timestep(full_window[occlusion_window[1]]))
                               for idx, occlusion_window in zip(target_agent_indices, occlusion_windows)]
    simulation_dict["occlusion_target_coords"] = occlusion_target_coords

    valid_occlusion_patterns = [False] * len(target_agent_indices)
    trial = 0

    ego_point = None
    ego_buffer = None
    p1_area = None
    p1_triangles = None
    p1_visipolys = None
    p2_area = None
    p2_triangles = None
    occluders = None
    ego_visipoly = None

    while trial < 5 and not all(valid_occlusion_patterns):
        # produce an ego_point from yes_ego_triangles
        ego_point = poly_op.random_points_in_triangle(*poly_op.sample_triangles(yes_ego_triangles, k=1), k=1).reshape(2)

        # draw circle around the ego_point
        ego_buffer = type_conv.shapely_poly_2_skgeom_poly(sp.Point(*ego_point).buffer(d_min_occl_ego))

        # ITERATE OVER TARGET AGENTS
        p1_area = []
        p1_triangles = []
        p1_visipolys = []
        p2_area = []
        p2_triangles = []
        occluders = []

        for idx, occlusion_window in zip(target_agent_indices, occlusion_windows):
            # triangle defined by ego, and the trajectory segment [t_occl: t_occl+1] of the target agent
            p1_ego_traj_triangle = sg.Polygon(np.array(
                [ego_point,
                 agents[idx].position_at_timestep(full_window[occlusion_window[0]]),
                 agents[idx].position_at_timestep(full_window[occlusion_window[0] + 1])]
            ))
            if p1_ego_traj_triangle.orientation() == sg.Sign.CLOCKWISE:
                p1_ego_traj_triangle.reverse_orientation()

            p1_area.append(p1_ego_traj_triangle)

            # extrude no_occluder_regions from the triangle
            p1_ego_traj_triangle = sg.PolygonSet(p1_ego_traj_triangle).difference(
                sg.PolygonSet(no_occluder_buffers + [ego_buffer]))

            # triangulate the resulting region
            p1_triangles = poly_op.triangulate_polyset(p1_ego_traj_triangle)
            p1_triangles.extend(p1_triangles)

            # sample our first occluder wall coordinate from the region
            p1 = poly_op.random_points_in_triangle(*poly_op.sample_triangles(p1_triangles, k=1), k=1)

            # compute the visibility polygon of this point (corresponds to the regions in space that can be linked to
            # the point with a straight line
            no_occl_segments = list(scene_boundary.edges)
            [no_occl_segments.extend(poly.edges) for poly in no_occluder_buffers + [ego_buffer]]
            visi_occl_arr = sg.arrangement.Arrangement()
            [visi_occl_arr.insert(seg) for seg in no_occl_segments]

            p1_visipoly = visibility.visibility_polygon(ego_point=p1, arrangement=visi_occl_arr)
            p1_visipolys.append(p1_visipoly)

            p2_ego_traj_triangle = sg.Polygon(np.array(
                [ego_point,
                 agents[idx].position_at_timestep(full_window[occlusion_window[1]]),
                 agents[idx].position_at_timestep(full_window[occlusion_window[1] - 1])]
            ))
            if p2_ego_traj_triangle.orientation() == sg.Sign.CLOCKWISE:
                p2_ego_traj_triangle.reverse_orientation()

            p2_area.append(p2_ego_traj_triangle)

            p2_ego_traj_triangle = sg.PolygonSet(p2_ego_traj_triangle).intersection(p1_visipoly)

            p2_triangles = poly_op.triangulate_polyset(p2_ego_traj_triangle)
            p2_triangles.extend(p2_triangles)

            p2 = poly_op.random_points_in_triangle(poly_op.sample_triangles(p2_triangles, k=1)[0], k=1)

            occluders.append((p1.flatten(), p2.flatten()))

        ego_visi_arrangement = sg.arrangement.Arrangement()
        [ego_visi_arrangement.insert(sg.Segment2(sg.Point2(*occluder_coords[0]), sg.Point2(*occluder_coords[1])))
         for occluder_coords in occluders]
        [ego_visi_arrangement.insert(segment) for segment in list(scene_boundary.edges)]

        ego_visipoly = visibility.visibility_polygon(ego_point=ego_point, arrangement=ego_visi_arrangement)

        # verify we do obtain the desired observable -> occluded -> observable pattern
        valid_occlusion_patterns = agent_op.verify_target_agents_occlusion_pattern(
            visibility_polygon=ego_visipoly,
            full_window=full_window[:-1],
            agents=agents,
            target_agent_indices=target_agent_indices,
            occlusion_windows=occlusion_windows
        )
        trial += 1

    if not all(valid_occlusion_patterns):
        raise AssertionError("occlusion pattern incorrect")

    occluded_regions = sg.PolygonSet(scene_boundary).difference(ego_visipoly)

    simulation_dict["ego_point"] = ego_point
    simulation_dict["ego_buffer"] = ego_buffer
    simulation_dict["p1_area"] = p1_area
    simulation_dict["p1_triangles"] = p1_triangles
    simulation_dict["p1_visipolys"] = p1_visipolys
    simulation_dict["p2_area"] = p2_area
    simulation_dict["p2_triangles"] = p2_triangles
    simulation_dict["occluders"] = occluders
    simulation_dict["ego_visipoly"] = ego_visipoly
    simulation_dict["occluded_regions"] = occluded_regions

    return simulation_dict


def runsim_on_entire_dataset(
        dataset_cfg: str,
        simulator_cfg: str
) -> None:
    import json
    import os.path
    import pickle
    import pandas as pd
    import logging
    from tqdm import tqdm

    dataset_config = conf.get_config(dataset_cfg)
    simulator_config = conf.get_config(simulator_cfg)

    # setting the random seed (for reproducibility)
    np.random.seed(simulator_config["rng_seed"])

    dataset = StanfordDroneDataset(config=dataset_config, split=None)
    sim_id = simulator_config['sim_id']

    # preparing the directory for saving simulation outputs
    pickle_path = os.path.abspath(os.path.join(conf.REPO_ROOT, 'outputs', 'pickled_dataloaders'))
    assert os.path.exists(pickle_path)

    sim_folder = os.path.join(pickle_path, dataset.pickle_id, sim_id)
    assert not os.path.exists(sim_folder), f"ERROR: target directory already exists:\n{sim_folder}"
    print(f"Creating simulation directory:\n{sim_folder}")
    os.makedirs(sim_folder)
    assert os.path.exists(sim_folder)

    pkl_path = os.path.join(sim_folder, "simulation.pickle")
    json_path = os.path.join(sim_folder, "simulation_parameters.json")
    log_path = os.path.join(sim_folder, "simulation.log")
    occl_path = os.path.join(sim_folder, "simulation_occlusions.pickle")

    # setting up the logger for traceback information of simulation failures
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(log_path, mode="a")
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)

    print(f"Saving simulation config to:\n{json_path}")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(simulator_config, f, indent=4)

    n_sim_per_instance = simulator_config["simulations_per_instance"]
    n_instances = len(dataset)
    print(f"\nRunning Simulator {n_sim_per_instance} times over {n_instances} individual instances\n")
    occlusion_df = pd.DataFrame(
        columns=["scene", "video", "timestep", "trial", "ego_point",
                 "occluders", "target_agent_indices", "occlusion_windows"]
    )
    occlusion_masks = []

    errors = 0
    for idx in (pbar := tqdm(range(n_instances))):

        pbar.set_description(f"ERRORS: {errors}")

        instance_dict = dataset.__getitem__(idx)

        scene = instance_dict["scene"]
        video = instance_dict["video"]
        timestep = instance_dict["timestep"]
        img = instance_dict["scene_image"]
        agents = instance_dict["agents"]
        past_window = instance_dict["past_window"]
        future_window = instance_dict["future_window"]

        for trial in range(n_sim_per_instance):
            try:
                simdict = simulate_occlusions(
                    config=simulator_config,
                    image_res=tuple(img.shape[:2]),
                    agents=agents,
                    past_window=past_window,
                    future_window=future_window
                )

                occlusion_df.loc[len(occlusion_df)] = {
                    "scene": scene,
                    "video": video,
                    "timestep": timestep,
                    "trial": trial,
                    "ego_point": simdict["ego_point"],
                    "occluders": simdict["occluders"],
                    "target_agent_indices": simdict["target_agent_indices"],
                    "occlusion_windows": simdict["occlusion_windows"]
                }

                occlusion_masks.append(visibility.agent_occlusion_masks(
                    agents=agents,
                    time_window=instance_dict["full_window"],
                    ego_visipoly=simdict["ego_visipoly"]
                ))

            except Exception as e:
                errors += 1
                logger.exception(f"\ninstance nr {idx} - trial nr {trial}:\n")

        if idx % 1000 == 0:
            print(f"Saving simulation table to:\n{pkl_path}")
            with open(pkl_path, "wb") as f:
                pickle.dump(occlusion_df, f)
            print(f"Saving occlusion masks to:\n{occl_path}")
            with open(occl_path, "wb") as f:
                pickle.dump(occlusion_masks, f)

    end_msg = f"\n\nTOTAL NUMBER OF ERRORS: {errors} ({errors/(n_instances * n_sim_per_instance)*100}%)\n"
    print(end_msg)
    logger.info(end_msg)

    # setting the indices for easy lookup, and sorting the dataframe
    occlusion_df.set_index(["scene", "video", "timestep", "trial"], inplace=True)
    occlusion_df.sort_index(inplace=True)

    print(f"Saving simulation table to:\n{pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump(occlusion_df, f)
    print(f"Saving occlusion masks to:\n{occl_path}")
    with open(occl_path, "wb") as f:
        pickle.dump(occlusion_masks, f)


def time_polygon_generation(instance_dict: dict, n_iterations: int = 1000000):
    from time import time
    print(f"Checking polygon generation timing: {n_iterations} iterations\n")
    before = time()
    for i in range(n_iterations):
        sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 1), sg.Point2(1, 1), sg.Point2(1, 0)])
    print(f"skgeom polygon instantiation: {time() - before}")

    before = time()
    for i in range(n_iterations):
        sp.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    print(f"shapely polygon instantiation: {time() - before}")

    before = time()
    polysg = sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 1), sg.Point2(1, 1), sg.Point2(1, 0)])
    polysg = sg.PolygonWithHoles(polysg, [])
    for i in range(n_iterations):
        type_conv.skgeom_poly_2_shapely_poly(polysg)
    print(f"skgeom 2 shapely polygon conversion: {time() - before}")

    before = time()
    polysp = sp.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    for i in range(n_iterations):
        type_conv.shapely_poly_2_skgeom_poly(polysp)
    print(f"shapely 2 skgeom polygon conversion: {time() - before}")

    before = time()
    for i in range(n_iterations):
        poly_gen.default_rectangle(instance_dict["scene_image"].shape[:2])
    print(f"default rectangle: {time() - before}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-cfg',
        type=os.path.abspath, default=os.path.join(conf.REPO_ROOT, 'config', 'dataset_config.yaml'),
        help='name of the .yaml config file to use for the parameters of the base SDD dataset.'
    )
    parser.add_argument(
        '--simulator-cfg',
        type=os.path.abspath, required=True,
        help='name of the .yaml config file to use for the parameters of occlusion simulator.'
    )
    args = parser.parse_args()

    runsim_on_entire_dataset(
        dataset_cfg=args.dataset_cfg,
        simulator_cfg=args.simulator_cfg
    )
