# import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import visualization.sdd_visualize as sdd_visualize
import visualization.simulation_visualize as sim_visualize
from shapely.geometry import Point, LineString, Polygon, GeometryCollection
from shapely.ops import triangulate, voronoi_diagram
from typing import List, Tuple, Union
import skgeom as sg
import functools
import itertools
from time import time
from scipy.interpolate import interp1d
import torch


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


def polygon_triangulate(polygon: Polygon) -> List[Polygon]:
    """
    'Naïve' polygon triangulation of the input. The triangulate function from shapely.ops does not guarantee proper
    triangulation of non-convex polygons with interior holes. This method permits this guarantee by performing
    triangulation on the union of points belonging to the polygon, and the points of the polygon's voronoi diagram.
    """
    voronoi_edges = voronoi_diagram(polygon, edges=True).intersection(polygon)
    # delaunay triangulation of every point (both from voronoi diagram and the polygon itself)
    candidate_triangles = triangulate(GeometryCollection([voronoi_edges, polygon]))
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


def select_random_target_agents(agent_list: List[sdd_dataloader.StanfordDroneAgent], past_window: np.array, n: int = 1)\
        -> List[int]:
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


def shapely_poly_2_skgeom_poly(poly: Polygon) -> sg.Polygon:
    return sg.Polygon([sg.Point2(*coord) for coord in poly.exterior.coords[:-1]][::-1])


def skgeom_poly_2_shapely_poly(poly: sg.PolygonWithHoles) -> Polygon:
    return Polygon(shell=poly.outer_boundary().coords, holes=[hole.coords for hole in poly.holes])


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
        agent_list: List[sdd_dataloader.StanfordDroneAgent],
        time_window: Union[None, np.array],
        buffer_radius: float
):
    return [
        shapely_poly_2_skgeom_poly(LineString(agent.get_traj_section(time_window)).buffer(buffer_radius))
        for agent in agent_list
    ]


def generate_occlusion_timesteps(n_agents: int, T_obs: int, T_pred: int, min_obs: int, min_reobs: int) -> List[Tuple[int, int]]:
    occlusion_windows = []
    for agent in range(n_agents):
        t_occl = np.random.randint(min_obs - 1, T_obs - 1)
        t_disoccl = np.random.randint(T_pred - min_reobs + 1) + T_obs
        occlusion_windows.append((t_occl, t_disoccl))
    return occlusion_windows


def trajectory_visibility_polygons(
        agents: List[sdd_dataloader.StanfordDroneAgent],
        target_agent_indices: List[int],
        agent_visipoly_buffers: List[sg.Polygon],
        time_window: np.array,
        boundary: sg.Polygon
) -> List[sg.PolygonSet]:
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


def perform_simulation(
        instance_dict: dict,
        image_tensor: torch.Tensor,
        agents: List[sdd_dataloader.StanfordDroneAgent],
        past_window: np.array,
        future_window: np.array
):
    n_targets = 2           # [-]   number of desired target agents to occlude virtually
    # n_egos = 1              # [-]   number of candidate positions to sample for the simulated ego

    min_obs = 4             # [-]   minimum amount of timesteps we want to have observed within observation window
    min_reobs = 2           # [-]   minimum amount of timesteps we want to be able to reobserve after disocclusion

    d_border = 200          # [px]  distance from scene border
    d_min_occl_ag = 60      # [px]  minimum distance that any point of a virtual occluder may have wrt any agent
    d_min_occl_ego = 30     # [px]  minimum distance that any point of a virtual occluder may have wrt ego
    d_min_ag_ego = (d_min_occl_ego + d_min_occl_ag) * 1.1

    taper_angle = 45        # [deg] angle for the generation of wedges

    r_agents = 10           # [px]  how "wide" we approximate agents to be

    full_window = np.concatenate((past_window, future_window))

    # set safety perimeter around the edges of the scene
    scene_boundary = default_rectangle(image_tensor.shape[1:])
    frame_box = skgeom_extruded_polygon(scene_boundary, d_border=d_border)

    # define agent_buffers, a list of sg.Polygons
    # corresponding to the past trajectories of every agent, inflated by some small radius (used for computation of
    # visibility polygons, in order to place the ego_point)
    agent_visipoly_buffers = trajectory_buffers(agents, past_window, r_agents)

    # define no_ego_buffers, a list of sg.Polygons, within which we wish not to place the ego
    no_ego_buffers = trajectory_buffers(agents, full_window, d_min_ag_ego)
    no_ego_buffers = sg.PolygonSet(no_ego_buffers)

    # define no_occluder_zones, a list of sg.Polygons, within which we wish not to place any virtual occluder
    no_occluder_buffers = trajectory_buffers(agents, full_window, d_min_occl_ag)
    # no_occluder_buffers = sg.PolygonSet(no_occluder_buffers)

    # choose agents within the scene whose trajectory we would like to occlude virtually
    target_agent_indices = select_random_target_agents(agents, past_window, n_targets)

    # define no_ego_wedges, a sg.PolygonSet, containing sg.Polygons within which we wish not to place the ego
    # the wedges are placed at the extremeties of the target agents, in order to prevent ego placements directly aligned
    # with the target agents' trajectories
    no_ego_wedges = sg.PolygonSet(list(itertools.chain(*[
        target_agent_no_ego_wedges(scene_boundary, agents[idx].get_traj_section(full_window), d_min_ag_ego, taper_angle)
        for idx in target_agent_indices
    ])))

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

    # list: a given item is a sg.PolygonSet, which describes the regions in space from which
    # every timestep of that agent can be directly observed, unobstructed by other agents
    # (specifically, by their agent_buffer)
    # target_agents_fully_observable_regions = []
    target_agents_fully_observable_regions = trajectory_visibility_polygons(
        agents=agents,
        target_agent_indices=target_agent_indices,
        agent_visipoly_buffers=agent_visipoly_buffers,
        time_window=full_window,
        boundary=scene_boundary
    )

    target_agents_fully_observable_regions = functools.reduce(
        lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
        target_agents_fully_observable_regions
    )

    # the regions within which we sample our ego are the regions within which target agents' full trajectories are
    # observable, minus the boundaries and no_ego_zones we set previously
    yes_ego_zones = target_agents_fully_observable_regions.difference(
        no_ego_buffers
    ).difference(
        frame_box
    ).difference(
        no_ego_wedges
    )

    # to sample from yes_ego_zones, we will need to triangulate the regions in yes_ego_zones
    # this can't be done in scikit-geometry (maybe it can?), so we're doing it with shapely instead
    yes_triangles = []
    [yes_triangles.extend(polygon_triangulate(skgeom_poly_2_shapely_poly(zone))) for zone in yes_ego_zones.polygons]
    yes_triangles = [shapely_poly_2_skgeom_poly(triangle) for triangle in yes_triangles]

    # sample points from yes_triangles
    ego_point = np.array(
        [random_points_in_triangle(triangle, k=1) for triangle in sample_triangles(yes_triangles, k=1)]
    ).reshape(2)

    # COMPUTE OCCLUDERS
    # draw circle around the ego_point
    ego_buffer = shapely_poly_2_skgeom_poly(Point(*ego_point).buffer(d_min_occl_ego))

    # ITERATE OVER TARGET AGENTS
    p1_ego_traj_triangles = []
    triangulated_p1_regions = []
    p1s = []
    p1_visipolys = []
    p2_ego_traj_triangles = []
    triangulated_p2_regions = []
    p2s = []
    occluder_segments = []

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

        p1_ego_traj_triangles.append(p1_ego_traj_triangle)

        # extrude no_occluder_regions from the triangle
        p1_ego_traj_triangle = sg.PolygonSet(p1_ego_traj_triangle).difference(
            sg.PolygonSet(no_occluder_buffers + [ego_buffer]))

        # triangulate the resulting region
        p1_triangles = []
        [p1_triangles.extend(
            polygon_triangulate(skgeom_poly_2_shapely_poly(poly))
        ) for poly in p1_ego_traj_triangle.polygons]
        p1_triangles = [shapely_poly_2_skgeom_poly(triangle) for triangle in p1_triangles]
        triangulated_p1_regions.extend(p1_triangles)

        # sample our first occluder wall coordinate from the region
        p1 = random_points_in_triangle(sample_triangles(p1_triangles, k=1)[0], k=1)
        p1s.append(p1)

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

        p2_ego_traj_triangles.append(p2_ego_traj_triangle)

        p2_ego_traj_triangle = sg.PolygonSet(p2_ego_traj_triangle).intersection(p1_visipoly)

        p2_triangles = []
        [p2_triangles.extend(
            polygon_triangulate(skgeom_poly_2_shapely_poly(poly))
        ) for poly in p2_ego_traj_triangle.polygons]
        p2_triangles = [shapely_poly_2_skgeom_poly(triangle) for triangle in p2_triangles]
        triangulated_p2_regions.extend(p2_triangles)

        p2 = random_points_in_triangle(sample_triangles(p2_triangles, k=1)[0], k=1)
        p2s.append(p2)

        occluder_seg = sg.Segment2(sg.Point2(*p1), sg.Point2(*p2))
        occluder_segments.append(occluder_seg)

    ego_visi_arrangement = sg.arrangement.Arrangement()
    [ego_visi_arrangement.insert(seg) for seg in occluder_segments + list(scene_boundary.edges)]

    ego_visipoly = visibility_polygon(ego_point=ego_point, arrangement=ego_visi_arrangement)
    occluded_regions = sg.PolygonSet(scene_boundary).difference(ego_visipoly)

    # visualization part
    p_occls = [agents[idx].position_at_timestep(full_window[occlusion_window[0]])
               for idx, occlusion_window in zip(target_agent_indices, occlusion_windows)]
    p_disoccls = [agents[idx].position_at_timestep(full_window[occlusion_window[1]])
                  for idx, occlusion_window in zip(target_agent_indices, occlusion_windows)]

    fig, axs = plt.subplots(nrows=2, ncols=3)
    [sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict) for ax in axs.reshape(-1)]
    sim_visualize.plot_simulation_step_1(axs[0, 0], agent_visipoly_buffers, no_occluder_buffers, no_ego_buffers,
                                         frame_box)
    sim_visualize.plot_simulation_step_2(axs[0, 1], agent_visipoly_buffers, no_ego_buffers, frame_box, no_ego_wedges,
                                         target_agents_fully_observable_regions, p_occls, p_disoccls)
    sim_visualize.plot_simulation_step_3(axs[0, 2], yes_triangles, p_occls, p_disoccls, ego_point, no_occluder_buffers,
                                         ego_buffer, p1_ego_traj_triangles)
    sim_visualize.plot_simulation_step_4(axs[1, 0], p_occls, p_disoccls, ego_point, triangulated_p1_regions, p1s,
                                         p1_visipolys, p2_ego_traj_triangles)
    sim_visualize.plot_simulation_step_5(axs[1, 1], p_occls, p_disoccls, ego_point, triangulated_p2_regions, p1s, p2s)
    sim_visualize.plot_simulation_step_6(axs[1, 2], p_occls, p_disoccls, ego_point, p1s, p2s, occluded_regions)

    # # I. agent buffers & frame_box
    # fig1, ax1 = plt.subplots()
    # sdd_visualize.visualize_training_instance(ax1, instance_dict=instance_dict)
    # sim_visualize.plot_simulation_step_1(ax1, agent_visipoly_buffers, no_occluder_buffers, no_ego_buffers, frame_box)
    #
    # # II. target agents' occlusion timesteps, wedges and visibility polygons
    # fig2, ax2 = plt.subplots()
    # sdd_visualize.visualize_training_instance(ax2, instance_dict=instance_dict)
    # sim_visualize.plot_simulation_step_2(ax2, agent_visipoly_buffers, no_ego_buffers, frame_box, no_ego_wedges,
    #                                      target_agents_fully_observable_regions, p_occls, p_disoccls)
    #
    # # III. triangulated ego regions, ego point, ego buffer, p1_ego_traj triangles
    # fig3, ax3 = plt.subplots()
    # sdd_visualize.visualize_training_instance(ax3, instance_dict=instance_dict)
    # sim_visualize.plot_simulation_step_3(ax3, yes_triangles, p_occls, p_disoccls, ego_point, no_occluder_buffers,
    #                                      ego_buffer, p1_ego_traj_triangles)
    #
    # # IV. triangulated p1_regions, p1, p1 visibility polygon, p2_ego_traj triangles
    # fig4, ax4 = plt.subplots()
    # sdd_visualize.visualize_training_instance(ax4, instance_dict=instance_dict)
    # sim_visualize.plot_simulation_step_4(ax4, p_occls, p_disoccls, ego_point, triangulated_p1_regions, p1s,
    #                                      p1_visipolys, p2_ego_traj_triangles)
    #
    # # V. triangulated p2_regions, p2
    # fig5, ax5 = plt.subplots()
    # sdd_visualize.visualize_training_instance(ax5, instance_dict=instance_dict)
    # sim_visualize.plot_simulation_step_5(ax5, p_occls, p_disoccls, ego_point, triangulated_p2_regions, p1s, p2s)
    #
    # # VI. occluder, ego_point visibility
    # fig6, ax6 = plt.subplots()
    # sdd_visualize.visualize_training_instance(ax6, instance_dict=instance_dict)
    # sim_visualize.plot_simulation_step_6(ax6, p_occls, p_disoccls, ego_point, p1s, p2s, occluded_regions)

    plt.show()


def time_polygon_generation(instance_dict: dict, n_iterations: int = 1000000):
    print(f"Checking polygon generation timing: {n_iterations} iterations\n")
    before = time()
    for i in range(n_iterations):
        sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 1), sg.Point2(1, 1), sg.Point2(1, 0)])
    print(f"skgeom polygon instantiation: {time() - before}")

    before = time()
    for i in range(n_iterations):
        Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    print(f"shapely polygon instantiation: {time() - before}")

    before = time()
    polysg = sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 1), sg.Point2(1, 1), sg.Point2(1, 0)])
    polysg = sg.PolygonWithHoles(polysg, [])
    for i in range(n_iterations):
        skgeom_poly_2_shapely_poly(polysg)
    print(f"skgeom 2 shapely polygon conversion: {time() - before}")

    before = time()
    polysp = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    for i in range(n_iterations):
        shapely_poly_2_skgeom_poly(polysp)
    print(f"shapely 2 skgeom polygon conversion: {time() - before}")

    # before = time()
    # for i in range(n_iterations):
    #     skgeom_rectangle(instance_dict["image_tensor"])
    # print(f"skgeom rectangle: {time() - before}")
    #
    # before = time()
    # for i in range(n_iterations):
    #     rectangle(instance_dict["image_tensor"])
    # print(f"shapely rectangle: {time() - before}")

    before = time()
    for i in range(n_iterations):
        default_rectangle(instance_dict["image_tensor"].shape[1:])
    print(f"default rectangle: {time() - before}")


def main():
    print("Ok, let's do this")
    instance_idx = 7592
    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    # time_polygon_generation(instance_dict=instance_dict, n_iterations=100000)
    img_tensor = instance_dict["image_tensor"]
    agents = instance_dict["agents"]
    past_window = instance_dict["past_window"]
    future_window = instance_dict["future_window"]

    perform_simulation(instance_dict, image_tensor=img_tensor, agents=agents, past_window=past_window, future_window=future_window)


if __name__ == '__main__':
    main()
