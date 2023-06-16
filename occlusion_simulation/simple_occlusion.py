import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as sp
import shapely.ops as spops
import skgeom as sg
import functools
import itertools
from scipy.interpolate import interp1d
from typing import List, Tuple, Union
import data.sdd_extract as sdd_extract
from data.sdd_dataloader import StanfordDroneDataset, StanfordDroneAgent
import visualization.simulation_visualize as sim_visualize
import visualization.sdd_visualize as sdd_visualize


# def point_between(point_1: np.array, point_2: np.array, k: float) -> np.array:
#     # k between 0 and 1, point_1 and point_2 of same shape
#     return k * point_1 + (1 - k) * point_2


def rotation_matrix(theta: float) -> np.array:
    # angle in radians
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def bounded_wedge(p: np.array, u: np.array, theta: float, boundary: sg.Polygon) -> sg.Polygon:
    """
    generates the polygon corresponding to a section of the plane bounded by:
    - a wedge described by two lines, intersecting at point p, such that their bisector points towards unit vector u.
    the angle of the lines with respect to the bisector is equal to theta.
    - frame_box, a polygon which contains point p.
    """
    big_vec = -u * 2 * np.max(boundary.coords)        # large vector, to guarantee cone is equivalent to infinite
    p1 = p + rotation_matrix(theta) @ big_vec         # CCW
    p2 = p + rotation_matrix(-theta) @ big_vec        # CW
    p3 = p1 + big_vec
    p4 = p2 + big_vec
    [out] = sg.boolean_set.intersect(sg.Polygon([p, p2, p4, p3, p1]), boundary)
    return out.outer_boundary()


def polygon_triangulate(polygon: sp.Polygon) -> List[sp.Polygon]:
    """
    'Naïve' polygon triangulation of the input. The triangulate function from shapely.ops does not guarantee proper
    triangulation of non-convex polygons with interior holes. This method permits this guarantee by performing
    triangulation on the union of points belonging to the polygon, and the points of the polygon's voronoi diagram.
    """
    voronoi_edges = spops.voronoi_diagram(polygon, edges=True).intersection(polygon)
    # delaunay triangulation of every point (both from voronoi diagram and the polygon itself)
    candidate_triangles = spops.triangulate(sp.GeometryCollection([voronoi_edges, polygon]))
    # keep only triangles inside original polygon
    return [triangle for triangle in candidate_triangles if triangle.centroid.within(polygon)]


def random_points_in_triangle(triangle: sg.Polygon, k: int = 1) -> np.array:
    # inspired by: https://stackoverflow.com/a/47418580
    x = np.sort(np.random.rand(2, k), axis=0)
    return triangle.coords.transpose() @ np.array([x[0], x[1]-x[0], 1.0-x[1]])


def random_points_in_triangles_collection(triangles: List[sg.Polygon], k: int) -> np.array:
    proportions = np.array([float(tri.area()) for tri in triangles])
    proportions /= sum(proportions)         # make a vector of probabilities
    triangles = [skgeom_poly_2_shapely_poly(sg.PolygonWithHoles(tri, [])) for tri in triangles]
    points = np.array(
        [random_points_in_triangle(triangles[idx]) for idx in
         np.random.choice(len(triangles), size=k, p=proportions)]
    ).reshape((k, 2))
    return points


def sample_triangles(triangles: List[sg.Polygon], k: int = 1) -> List[sg.Polygon]:
    proportions = np.array([float(tri.area()) for tri in triangles])
    proportions /= sum(proportions)         # make a vector of probabilities
    return [triangles[idx] for idx in np.random.choice(len(triangles), size=k, p=proportions)]


# def random_interpolated_point(traj_seq: np.array, timestep_bounds: Tuple[int, int]):
#     """
#     sample a random point among the interpolated sequence of coordinates traj_seq. the point will lie within the
#     interval specified by timestep_bounds (both inclusive)
#     :param traj_seq: shape [Timesteps, Dimensions]
#     :param timestep_bounds: tuple (begin, end)
#     """
#     t_idx = np.random.randint(timestep_bounds[0], timestep_bounds[1]-1)
#     return point_between(traj_seq[t_idx], traj_seq[t_idx+1], np.random.random())


def default_rectangle(corner_coords: Tuple[float, float]) -> sg.Polygon:
    """
    WARNING: having any of the two input values equal to 0.0 can result in errors
    :param corner_coords: tuple [height, width], as can be extracted from an image torch.Tensor of shape [C, H, W]
    """
    y, x = corner_coords
    return sg.Polygon([[0, 0], [x, 0], [x, y], [0, y]])


def skgeom_extruded_polygon(polygon: sg.Polygon, d_border: float) -> sg.PolygonWithHoles:
    skel = sg.skeleton.create_interior_straight_skeleton(polygon)
    return functools.reduce(lambda a, b: sg.boolean_set.difference(a, b)[0], skel.offset_polygons(d_border), polygon)


def select_random_target_agents(agent_list: List[StanfordDroneAgent], past_window: np.array, n: int = 1) -> List[int]:
    """
    selects a random subset of agents present within the scene. The probability to select any agent is proportional to
    the distance they travel. n agents will be sampled (possibly fewer if there aren't enough agents).
    the output provides the indices in agent_list pointing at our target agents.
    """
    # distances travelled by all agents in their past segment
    pasttrajs = [agent.get_traj_section(past_window) for agent in agent_list]
    distances = np.array([np.linalg.norm(pasttraj[-1] - pasttraj[0]) for pasttraj in pasttrajs])
    is_moving = (distances > 1e-8)

    candidate_indices = np.asarray(is_moving).nonzero()[0]
    distances = distances[is_moving]

    if sum(is_moving) == 0:
        print("Zero moving agents, no target agent can be selected")
        return []

    if sum(is_moving) <= n:
        print(f"returning all available candidates: only {sum(is_moving)} moving agents in the scene")
        return candidate_indices

    return list(np.random.choice(candidate_indices, n, replace=False, p=distances/sum(distances)))


def target_agent_no_ego_wedges(boundary: sg.Polygon, traj: np.array, offset: float, angle: float) -> List[sg.Polygon]:
    # offset: distance to pull the wedges "inward"
    u_traj = np.array(traj[-1] - traj[0])
    u_traj /= np.linalg.norm(u_traj)
    wedge_1 = bounded_wedge(
        p=(np.array(traj[0])) + u_traj * offset,
        u=u_traj,
        theta=float(np.radians(angle)),
        boundary=boundary
    )
    wedge_2 = bounded_wedge(
        p=(np.array(traj[-1])) - u_traj * offset,
        u=-u_traj,
        theta=float(np.radians(angle)),
        boundary=boundary
    )
    return [wedge_1, wedge_2]


def shapely_poly_2_skgeom_poly(poly: sp.Polygon) -> sg.Polygon:
    return sg.Polygon([sg.Point2(*coord) for coord in poly.exterior.coords[:-1]][::-1])


def skgeom_poly_2_shapely_poly(poly: sg.PolygonWithHoles) -> sp.Polygon:
    return sp.Polygon(shell=poly.outer_boundary().coords, holes=[hole.coords for hole in poly.holes])


def skgeom_approximate_circle(circ: sg.Circle2, n_segments: int = 100) -> sg.Polygon:
    thetas = np.linspace(0, 2 * np.pi, num=n_segments, endpoint=False)
    coords = [rotation_matrix(theta) @ np.array([circ.squared_radius(), 0]) +
              np.array([circ.center().x(), circ.center().y()]) for theta in thetas]
    return sg.Polygon([sg.Point2(*coord) for coord in coords])


def visibility_polygon(ego_point: Tuple[float, float], arrangement: sg.arrangement.Arrangement) -> sg.Polygon:
    visibility = sg.RotationalSweepVisibility(arrangement)
    # visibility = sg.TriangularExpansionVisibility(arrangement)
    origin = sg.Point2(*ego_point)
    face = arrangement.find(origin)
    vx = visibility.compute_visibility(origin, face)
    # convert Arrangement to sg.Polygon
    return sg.Polygon([pt.point() for pt in vx.vertices])


def trajectory_2_segment2_list(traj: np.array) -> List[sg.Segment2]:
    segments = []
    for idx in range(traj.shape[0] - 1):
        start = sg.Point2(traj[idx, 0], traj[idx, 1])
        end = sg.Point2(traj[idx + 1, 0], traj[idx + 1, 1])
        segments.append(sg.Segment2(start, end))
    return segments


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


def trajectory_buffers(
        agent_list: List[StanfordDroneAgent],
        time_window: Union[None, np.array],
        buffer_radius: float
) -> List[sg.Polygon]:
    if time_window is not None:
        return [
            shapely_poly_2_skgeom_poly(sp.LineString(agent.get_traj_section(time_window)).buffer(buffer_radius))
            for agent in agent_list
        ]
    return [
        shapely_poly_2_skgeom_poly(sp.LineString(agent.fulltraj).buffer(buffer_radius))
        for agent in agent_list
    ]


def generate_occlusion_timesteps(
        n_agents: int, T_obs: int, T_pred: int, min_obs: int, min_reobs: int
) -> List[Tuple[int, int]]:
    occlusion_windows = []
    for agent in range(n_agents):
        t_occl = np.random.randint(min_obs - 1, T_obs - 1)
        t_disoccl = np.random.randint(T_pred - min_reobs + 1) + T_obs
        occlusion_windows.append((t_occl, t_disoccl))
    return occlusion_windows


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
    all_trajs = [agent.get_traj_section(time_window) for agent in agents]
    interps_x = [interp1d(time_window, traj[:, 0]) for traj in all_trajs]
    interps_y = [interp1d(time_window, traj[:, 1]) for traj in all_trajs]
    interp_time = np.arange(time_window[0], time_window[-1], interp_dt)

    # print(time_window)
    # print(interp_time)
    # print(all_trajs[0])
    # print(interps_x[0])
    # print(interps_y[0])
    # print(interps_x[0](interp_time[4]))

    target_visipolys = []

    for idx in target_agent_indices:
        other_interps_x = interps_x.copy()
        other_interps_x.pop(idx)
        other_interps_y = interps_y.copy()
        other_interps_y.pop(idx)

        agent_visipoly = []

        for t in interp_time:

            scene_segments = list(boundary.edges)
            for interp_x, interp_y in zip(other_interps_x, other_interps_y):
                circ = skgeom_approximate_circle(
                    sg.Circle2(sg.Point2(interp_x(t), interp_y(t)), sg.FieldNumberType(agent_radius), sg.Sign(1))
                )
                scene_segments.extend(list(circ.edges))

            scene_arr = sg.arrangement.Arrangement()
            [scene_arr.insert(seg) for seg in scene_segments]

            agent_visipoly.append(sg.PolygonSet(visibility_polygon((interps_x[idx](t), interps_y[idx](t)), arrangement=scene_arr)))

        target_visipolys.append(functools.reduce(
            lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
            agent_visipoly
        ))

    return target_visipolys


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
            [sg.PolygonSet(visibility_polygon(point, arrangement=scene_arr))
             for point in interpolate_trajectory(traj, dt=10)]
        )
        trajectory_visipolys.append(traj_fully_observable)

    return trajectory_visipolys


def triangulate_polyset(polyset: sg.PolygonSet) -> List[sg.Polygon]:
    triangles = list(itertools.chain(*[
        polygon_triangulate(skgeom_poly_2_shapely_poly(poly)) for poly in polyset.polygons
    ]))
    return [shapely_poly_2_skgeom_poly(triangle) for triangle in triangles]


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


def simulate_occlusions(
        config: dict,
        image_res: Tuple[int, int],
        agents: List[StanfordDroneAgent],
        past_window: np.array,
        future_window: np.array
):
    n_targets = config["n_target_agents"]

    min_obs = config["min_obs"]
    min_reobs = config["min_reobs"]

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

    full_window = np.concatenate((past_window, future_window))

    # choose agents within the scene whose trajectory we would like to occlude virtually
    target_agent_indices = select_random_target_agents(agents, past_window, n_targets)
    simulation_dict["target_agent_indices"] = target_agent_indices

    # generate occlusion windows -> List[Tuple[int, int]]
    # each item provides two timesteps for each target agent:
    # - the first one corresponds to the last observed timestep before occlusion
    # - the second one corresponds to the first re-observed timestep before reappearance
    occlusion_windows = generate_occlusion_timesteps(
        n_agents=n_targets,
        T_obs=past_window.shape[0],
        T_pred=future_window.shape[0],
        min_obs=min_obs,
        min_reobs=min_reobs
    )
    simulation_dict["occlusion_windows"] = occlusion_windows

    # set safety perimeter around the edges of the scene
    d_border_px = int((d_border/100 * np.linalg.norm(image_res)) // 10 * 11)
    scene_boundary = default_rectangle(image_res)
    frame_box = skgeom_extruded_polygon(scene_boundary, d_border=d_border_px)
    simulation_dict["frame_box"] = frame_box

    # define agent_buffers, a list of sg.Polygons
    # corresponding to the past trajectories of every agent, inflated by some small radius (used for computation of
    # visibility polygons, in order to place the ego_point)
    agent_visipoly_buffers = trajectory_buffers(agents, past_window, r_agents)
    simulation_dict["agent_visipoly_buffers"] = agent_visipoly_buffers

    # define no_ego_buffers, a list of sg.Polygons, within which we wish not to place the ego
    no_ego_buffers = trajectory_buffers(agents, full_window, d_min_ag_ego)
    no_ego_buffers = sg.PolygonSet(no_ego_buffers)
    simulation_dict["no_ego_buffers"] = no_ego_buffers

    # define no_occluder_zones, a list of sg.Polygons, within which we wish not to place any virtual occluder
    no_occluder_buffers = trajectory_buffers(agents, full_window, d_min_occl_ag)
    simulation_dict["no_occluder_buffers"] = no_occluder_buffers

    # define no_ego_wedges, a sg.PolygonSet, containing sg.Polygons within which we wish not to place the ego
    # the wedges are placed at the extremeties of the target agents, in order to prevent ego placements directly aligned
    # with the target agents' trajectories
    no_ego_wedges = sg.PolygonSet(list(itertools.chain(*[
        target_agent_no_ego_wedges(scene_boundary, agents[idx].get_traj_section(full_window), d_min_ag_ego, taper_angle)
        for idx in target_agent_indices
    ])))
    simulation_dict["no_ego_wedges"] = no_ego_wedges

    # list: a given item is a sg.PolygonSet, which describes the regions in space from which
    # every timestep of that agent can be directly observed, unobstructed by other agents
    # (specifically, by their agent_buffer)
    targets_fullobs_regions = trajectory_visibility_polygons(
        agents=agents,
        target_agent_indices=target_agent_indices,
        agent_visipoly_buffers=agent_visipoly_buffers,
        time_window=full_window,
        boundary=scene_boundary
    )
    # targets_fullobs_regions = instantaneous_visibility_polygons(
    #     agents=agents,
    #     target_agent_indices=target_agent_indices,
    #     agent_radius=r_agents,
    #     interp_dt=1,
    #     time_window=past_window,
    #     boundary=scene_boundary
    # )

    # reducing into a single sg.PolygonSet
    targets_fullobs_regions = functools.reduce(
        lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
        targets_fullobs_regions
    )
    simulation_dict["targets_fullobs_regions"] = targets_fullobs_regions

    # the regions within which we sample our ego are those within which target agents' full trajectories are
    # observable, minus the boundaries and no_ego_zones we set previously.
    # we will need to triangulate those regions in order to sample a point
    # this can't be done in scikit-geometry (maybe it can?), so we're doing it with shapely instead
    # (see inside triangulate_polyset function)
    yes_ego_triangles = triangulate_polyset(
        targets_fullobs_regions.difference(
            no_ego_buffers.union(no_ego_wedges).union(frame_box)
        )
    )
    simulation_dict["yes_ego_triangles"] = yes_ego_triangles

    # produce an ego_point from yes_ego_triangles
    ego_point = random_points_in_triangle(*sample_triangles(yes_ego_triangles, k=1), k=1).reshape(2)
    simulation_dict["ego_point"] = ego_point

    # COMPUTE OCCLUDERS
    # draw circle around the ego_point
    ego_buffer = shapely_poly_2_skgeom_poly(sp.Point(*ego_point).buffer(d_min_occl_ego))
    simulation_dict["ego_buffer"] = ego_buffer

    # ITERATE OVER TARGET AGENTS
    p1_area = []
    p1_triangles = []
    p1_visipolys = []
    p2_area = []
    p2_triangles = []
    occluders = []

    for idx, occlusion_window in zip(target_agent_indices, occlusion_windows):
        # occlusion_window = (t_occl, t_disoccl)

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
        p1_triangles = triangulate_polyset(p1_ego_traj_triangle)
        p1_triangles.extend(p1_triangles)

        # sample our first occluder wall coordinate from the region
        p1 = random_points_in_triangle(*sample_triangles(p1_triangles, k=1), k=1)

        # compute the visibility polygon of this point (corresponds to the regions in space that can be linked to
        # the point with a straight line
        no_occl_segments = list(scene_boundary.edges)
        [no_occl_segments.extend(poly.edges) for poly in no_occluder_buffers + [ego_buffer]]
        visi_occl_arr = sg.arrangement.Arrangement()
        [visi_occl_arr.insert(seg) for seg in no_occl_segments]

        p1_visipoly = visibility_polygon(ego_point=p1, arrangement=visi_occl_arr)
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

        p2_triangles = triangulate_polyset(p2_ego_traj_triangle)
        p2_triangles.extend(p2_triangles)

        p2 = random_points_in_triangle(sample_triangles(p2_triangles, k=1)[0], k=1)

        occluders.append((p1, p2))

    simulation_dict["p1_area"] = p1_area
    simulation_dict["p1_triangles"] = p1_triangles
    simulation_dict["p1_visipolys"] = p1_visipolys
    simulation_dict["p2_area"] = p2_area
    simulation_dict["p2_triangles"] = p2_triangles
    simulation_dict["occluders"] = occluders

    ego_visi_arrangement = sg.arrangement.Arrangement()
    [ego_visi_arrangement.insert(sg.Segment2(sg.Point2(*occluder_coords[0]), sg.Point2(*occluder_coords[1])))
     for occluder_coords in occluders]
    [ego_visi_arrangement.insert(segment) for segment in list(scene_boundary.edges)]

    ego_visipoly = visibility_polygon(ego_point=ego_point, arrangement=ego_visi_arrangement)
    occluded_regions = sg.PolygonSet(scene_boundary).difference(ego_visipoly)
    simulation_dict["occluded_regions"] = occluded_regions

    # verify we do obtain the desired observable -> occluded -> observable pattern
    occlusion_patterns_correct = verify_target_agents_occlusion_pattern(
        visibility_polygon=ego_visipoly,
        full_window=full_window,
        agents=agents,
        target_agent_indices=target_agent_indices,
        occlusion_windows=occlusion_windows
    )
    if not all(occlusion_patterns_correct):
        raise AssertionError("occlusion pattern incorrect")

    # simulation_dict = {
    #     "target_agent_indices": target_agent_indices,
    #     "occlusion_windows": occlusion_windows,
    #     "frame_box": frame_box,
    #     "agent_visipoly_buffers": agent_visipoly_buffers,
    #     "no_occluder_buffers": no_occluder_buffers,
    #     "no_ego_buffers": no_ego_buffers,
    #     "no_ego_wedges": no_ego_wedges,
    #     "targets_fullobs_regions": targets_fullobs_regions,
    #     "yes_ego_triangles": yes_ego_triangles,
    #     "ego_point": ego_point,
    #     "ego_buffer": ego_buffer,
    #     "p1_area": p1_area,
    #     "p1_triangles": p1_triangles,
    #     "p1_visipolys": p1_visipolys,
    #     "p2_area": p2_area,
    #     "p2_triangles": p2_triangles,
    #     "occluders": occluders,
    #     "occluded_regions": occluded_regions
    # }

    return simulation_dict


def runsim_on_entire_dataset() -> None:
    import json
    import os.path
    import pickle
    import pandas as pd
    import logging
    from tqdm import tqdm

    config = sdd_extract.get_config()

    # setting the random seed (for reproducibility)
    np.random.seed(config["occlusion_simulator"]["rng_seed"])

    dataset = StanfordDroneDataset(config_dict=config)

    # save the simulation dataframe and configuration dictionary to appropriate pickle and json files
    sim_save_name = f"{dataset.pickle_id}_sim"
    pkl_path = os.path.join(config["dataset"]["pickle_path"], f"{sim_save_name}.pickle")
    json_path = os.path.join(config["dataset"]["pickle_path"], f"{sim_save_name}.json")
    log_path = os.path.join(config["dataset"]["pickle_path"], f"{sim_save_name}.log")

    # setting up the logger for traceback information of simulation failures
    logger = logging.getLogger(__name__)
    # c_handler = logging.StreamHandler()
    # c_handler.setLevel(logging.WARNING)
    # logger.addHandler(c_handler)
    f_handler = logging.FileHandler(log_path)
    f_handler.setLevel(logging.INFO)
    logger.addHandler(f_handler)

    n_sim_per_instance = config["occlusion_simulator"]["simulations_per_instance"]
    n_instances = len(dataset)
    n_instances = 1000
    print(f"\nRunning Simulator {n_sim_per_instance} times over {n_instances} individual instances\n")
    occlusion_df = pd.DataFrame(
        columns=["scene", "video", "timestep", "trial", "ego_point",
                 "occluders", "target_agent_indices", "occlusion_windows"]
    )

    value_errors = 0
    runtime_errors = 0
    assert_errors = 0
    other_errors = 0
    for idx in (pbar := tqdm(range(n_instances))):

        pbar.set_description(f"ERRORS: v: {value_errors}, r: {runtime_errors}, a: {assert_errors}, o: {other_errors}")

        instance_dict = dataset.__getitem__(idx)

        scene = instance_dict["scene"]
        video = instance_dict["video"]
        timestep = instance_dict["timestep"]
        img_tensor = instance_dict["image_tensor"]
        agents = instance_dict["agents"]
        past_window = instance_dict["past_window"]
        future_window = instance_dict["future_window"]

        for trial in range(n_sim_per_instance):
            try:
                simdict = simulate_occlusions(
                    config=config["occlusion_simulator"],
                    image_res=tuple(img_tensor.shape[:1]),
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
            except ValueError as e:
                value_errors += 1
                logger.exception(f"\ninstance nr {idx} - trial nr {trial}:\n")
            except RuntimeError as e:
                runtime_errors += 1
                logger.exception(f"\ninstance nr {idx} - trial nr {trial}:\n")
            except AssertionError as e:
                assert_errors += 1
                logger.exception(f"\ninstance nr {idx} - trial nr {trial}:\n")
            except Exception as e:
                other_errors += 1
                logger.exception(f"\ninstance nr {idx} - trial nr {trial}:\n")

    total_errors = value_errors + runtime_errors + assert_errors + other_errors
    end_msg = f"\n\nTOTAL NUMBER OF ERRORS: {total_errors} ({total_errors/(n_instances * n_sim_per_instance)*100}%)\n" \
              f"ValueError: {value_errors}\n" \
              f"RuntimeError: {runtime_errors}\n" \
              f"AssertionError: {assert_errors}\n" \
              f"Other: {other_errors}\n\n"
    print(end_msg)
    logger.info(end_msg)

    # setting the indices for easy lookup, and sorting the dataframe
    occlusion_df.set_index(["scene", "video", "timestep", "trial"], inplace=True)
    occlusion_df.sort_index(inplace=True)

    if os.path.exists(pkl_path):
        print(f"\nRemoving simulation pickle file (already exists):\n{pkl_path}\n")
        os.remove(pkl_path)
    if os.path.exists(json_path):
        print(f"\nRemoving simulation config file (already exists):\n{json_path}\n")
        os.remove(json_path)

    print(f"Saving simulation table to:\n{pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump((occlusion_df), f)

    print(f"Saving simulation config to:\n{json_path}")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(config["occlusion_simulator"], f, indent=4)


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
        skgeom_poly_2_shapely_poly(polysg)
    print(f"skgeom 2 shapely polygon conversion: {time() - before}")

    before = time()
    polysp = sp.Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    for i in range(n_iterations):
        shapely_poly_2_skgeom_poly(polysp)
    print(f"shapely 2 skgeom polygon conversion: {time() - before}")

    before = time()
    for i in range(n_iterations):
        default_rectangle(instance_dict["image_tensor"].shape[1:])
    print(f"default rectangle: {time() - before}")


def show_simulation():
    config = sdd_extract.get_config()
    dataset = StanfordDroneDataset(config_dict=config)

    # showing the simulation process of some desired instance
    instance_idx = 7592     # coupa video0 60
    # instance_idx = 36371    # nexus video7 3024
    # instance_idx = np.random.randint(len(dataset))
    print(f"dataset.__getitem__({instance_idx})")
    instance_dict = dataset.__getitem__(instance_idx)

    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(draw_ax=ax, instance_dict=instance_dict)

    # time_polygon_generation(instance_dict=instance_dict, n_iterations=100000)
    sim_params = config["occlusion_simulator"]
    img_tensor = instance_dict["image_tensor"]
    agents = instance_dict["agents"]
    past_window = instance_dict["past_window"]
    future_window = instance_dict["future_window"]

    simulation_outputs = simulate_occlusions(
        config=sim_params,
        image_res=tuple(img_tensor.shape[1:]),
        agents=agents,
        past_window=past_window,
        future_window=future_window
    )
    sim_visualize.visualize_occlusion_simulation(instance_dict, simulation_outputs)

    # Showing the simulation outputs of some random instances
    sim_visualize.visualize_random_simulation_samples(dataset, config["occlusion_simulator"], 2, 2)


if __name__ == '__main__':
    # show_simulation()
    runsim_on_entire_dataset()
