import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import data.sdd_visualize as sdd_visualize
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import matplotlib.collections as mpl_coll
from shapely.geometry import Point, LineString, Polygon, GeometryCollection, MultiPolygon
from shapely.ops import unary_union, triangulate, voronoi_diagram
from typing import List, Tuple, Union

import skgeom as sg
import functools
from time import time


def point_between(point_1: np.array, point_2: np.array, k: float) -> np.array:
    # k between 0 and 1, point_1 and point_2 of same shape
    return k * point_1 + (1 - k) * point_2


def rotation_matrix(theta: float) -> np.array:
    # angle in radians
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def bounded_wedge(p: np.array, u: np.array, theta: float, frame_box: Polygon) -> Polygon:
    """
    generates the polygon corresponding to a section of the plane bounded by:
    - a wedge described by two lines, intersecting at point p, such that their bisector points towards unit vector u.
    the angle of the lines with respect to the bisector is equal to theta.
    - frame_box, a polygon which contains point p.
    """
    big_vec = -u * 2 * np.max(frame_box.bounds)        # large vector, to guarantee cone is equivalent to infinite
    p1 = rotation_matrix(theta) @ big_vec + p
    p2 = rotation_matrix(-theta) @ big_vec + p
    p3 = p1 + big_vec
    p4 = p2 + big_vec
    return Polygon([p, p1, p3, p4, p2, p]).intersection(frame_box)


def skgeom_bounded_wedge(p: np.array, u: np.array, theta: float, frame_box: Polygon) -> Polygon:
    # todo OR NOT TO DO
    pass


def plot_polygon(ax: matplotlib.axes.Axes, poly: Polygon, **kwargs) -> None:
    path = mpl_path.Path.make_compound_path(
        mpl_path.Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[mpl_path.Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors]
    )

    patch = mpl_patches.PathPatch(path, **kwargs)
    collection = mpl_coll.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()


def skgeom_plot_polygon(ax: matplotlib.axes.Axes, poly: Union[sg.Polygon, sg.PolygonWithHoles], **kwargs) -> None:
    if isinstance(poly, sg.Polygon):
        path = mpl_path.Path(poly.coords)
        # coords = np.concatenate([poly.coords, [poly.coords[-1]]])
        # path = mpl_path.Path(coords)
    elif isinstance(poly, sg.PolygonWithHoles):
        path = mpl_path.Path.make_compound_path(
            mpl_path.Path(poly.outer_boundary().coords),
            *[mpl_path.Path(hole.coords) for hole in poly.holes]
        )

    patch = mpl_patches.PathPatch(path, **kwargs)
    collection = mpl_coll.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()


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


def skgeom_polygon_triangulate(polygon: Polygon) -> List[Polygon]:
    # todo OR NOT TO DO
    pass


def random_points_in_triangle(triangle: Polygon, k: int = 1) -> np.array:
    # inspired by: https://stackoverflow.com/a/47418580
    x = np.sort(np.random.rand(2, k), axis=0)
    return np.array(triangle.exterior.xy)[:, :-1] @ np.array([x[0], x[1]-x[0], 1.0-x[1]])


def skgeom_random_points_in_triangle(triangle: Polygon, k: int = 1) -> np.array:
    # todo OR NOT TO DO
    pass


def random_points_in_triangles_collection(triangles: MultiPolygon, k: int) -> np.array:
    proportions = np.array([tri.area for tri in triangles.geoms])
    proportions /= sum(proportions)         # make a vector of probabilities
    points = np.array(
        [random_points_in_triangle(triangles.geoms[idx]) for idx in
         np.random.choice(len(triangles.geoms), size=k, p=proportions)]
    ).reshape((k, 2))
    return points


def random_interpolated_point(traj_seq: np.array, timestep_bounds: Tuple[int, int]):
    """
    sample a random point among the interpolated sequence of coordinates traj_seq. the point will lie within the
    interval specified by timestep_bounds (both inclusive)
    :param traj_seq: shape [Timesteps, Dimensions]
    :param timestep_bounds: tuple (begin, end)
    """
    t_idx = np.random.randint(timestep_bounds[0], timestep_bounds[1]-1)
    return point_between(traj_seq[t_idx], traj_seq[t_idx+1], np.random.random())


def rectangle(image_tensor: torch.Tensor) -> Polygon:
    # returns a rectangle with dimensions corresponding to those of the input image_tensor
    y_img, x_img = image_tensor.shape[1:]
    return Polygon([(0, 0), (x_img, 0), (x_img, y_img), (0, y_img), (0, 0)])


def skgeom_rectangle(image_tensor: torch.Tensor):
    y_img, x_img = image_tensor.shape[1:]
    return sg.Polygon([[0, 0], [x_img, 0], [x_img, y_img], [0, y_img]])


def extruded_polygon(polygon: Polygon, d_border: float) -> Polygon:
    hole = polygon.buffer(-d_border)
    return Polygon(shell=polygon.exterior.coords, holes=[hole.exterior.coords])


def skgeom_extruded_polygon(polygon: sg.Polygon, d_border: float) -> sg.PolygonWithHoles:
    skel = sg.skeleton.create_interior_straight_skeleton(polygon)
    return functools.reduce(lambda a, b: sg.boolean_set.difference(a, b)[0], skel.offset_polygons(d_border), polygon)


def select_random_target_agents(instance_dict: dict, n: int = 1) -> List[int]:
    """
    selects a random subset of agents present within the scene. The probability to select any agent is proportional to
    the distance they travel. n agents will be sampled (possibly fewer if there aren't enough agents)
    """
    # distances travelled by all agents in their past segment
    distances = np.array([np.linalg.norm(pasttraj[-1] - pasttraj[0]) for pasttraj in instance_dict["pasts"]])
    is_moving = (distances > 1e-8)

    # keeping the agents which have travelled a nonzero distance in their past
    ids = np.array(instance_dict["agent_ids"])[is_moving]
    distances = distances[is_moving]

    if ids.size == 0:
        print("Zero moving agents, no target agent can be selected")
        return []

    if ids.size <= n:
        print(f"returning all available candidates: only {ids.size} moving agents in the scene")
        return list(ids)

    return list(np.random.choice(ids, n, replace=False, p=distances/sum(distances)))


def target_agent_no_ego_zones(fulltraj: np.array, radius: float = 60, wedge_angle: float = 60) -> List[Polygon]:
    # generate safety buffer area around agent's trajectory
    target_buffer = LineString(fulltraj).buffer(radius).convex_hull

    # define no-ego wedges based on taper angle, in order to prevent situations where ego
    # is directly aligned with target agent's trajectory
    u_traj = np.array(fulltraj[-1] - fulltraj[0])  # unit vector of agent's trajectory (future[-1], past[0])
    u_traj /= np.linalg.norm(u_traj)
    no_ego_1 = bounded_wedge((np.array(fulltraj[-1]) - u_traj * radius), -u_traj, float(np.radians(wedge_angle)),
                             rectangle(instance_dict["image_tensor"]))
    no_ego_2 = bounded_wedge((np.array(fulltraj[0]) + u_traj * radius), u_traj, float(np.radians(wedge_angle)),
                             rectangle(instance_dict["image_tensor"]))

    return [target_buffer, no_ego_1, no_ego_2]


def skgeom_target_agent_no_ego_zones(fulltraj: np.array, radius: float = 60, wedge_angle: float = 60) -> List[sg.Polygon]:
    # todo OR NOT TODO
    pass


def shapely_poly_2_skgeom_poly(poly: Polygon) -> Union[sg.Polygon, sg.PolygonWithHoles]:
    # TODO: ENABLE WORKING WITH POLY WITH HOLES
    return sg.Polygon([sg.Point2(*coord) for coord in poly.exterior.coords[:-1]])


def skgeom_poly_2_shapely_poly(poly: Union[sg.Polygon, sg.PolygonWithHoles]) -> Polygon:
    # TODO: ENABLE WORKING WITH POLY WITH HOLES
    return Polygon(poly.coords)


def skgeom_approximate_circle(circ: sg.Circle2, n_segments: int = 100) -> sg.Polygon:
    thetas = np.linspace(0, 2 * np.pi, num=n_segments, endpoint=False)
    coords = [rotation_matrix(theta) @ np.array([circ.squared_radius(), 0]) +
              np.array([circ.center().x(), circ.center().y()]) for theta in thetas]
    return sg.Polygon([sg.Point2(*coord) for coord in coords])


def visibility_polygon(ego_point: np.array, segments):
    arr = sg.arrangement.Arrangement()
    [arr.insert(seg) for seg in segments]

    visibility = sg.RotationalSweepVisibility(arr)
    # visibility = sg.TriangularExpansionVisibility(arr)
    origin = sg.Point2(*ego_point)
    face = arr.find(origin)
    vx = visibility.compute_visibility(origin, face)

    # convert Arrangement to sg.Polygon
    vx = sg.Polygon([pt.point() for pt in vx.vertices])
    return vx


def trajectory_2_segment2_list(traj: np.array) -> List[sg.Segment2]:
    segments = []
    for idx in range(traj.shape[0] - 1):
        start = sg.Point2(traj[idx, 0], traj[idx, 1])
        end = sg.Point2(traj[idx + 1, 0], traj[idx + 1, 1])
        segments.append(sg.Segment2(start, end))
    return segments


def place_ego(instance_dict: dict):
    n_targets = 1           # [-]   number of desired target agents to occlude virtually
    d_border = 200          # [px]  distance from scene border
    min_obs = 2             # [-]   minimum amount of timesteps we want to have observed within observation window
    min_reobs = 2           # [-]   minimum amount of timesteps we want to be able to reobserve after disocclusion
    d_min_occl_ag = 60      # [px]  minimum distance that any point of a virtual occluder may have wrt any agent
    d_min_occl_ego = 30     # [px]  minimum distance that any point of a virtual occluder may have wrt ego
    r_agents = 10           # [px]  how "wide" we approximate agents to be
    taper_angle = 45        # [deg] angle for the generation of wedges
    n_egos = 1              # [-]   number of candidate positions to sample for the simulated ego

    target_agents = select_random_target_agents(instance_dict, n_targets)

    # set safety perimeter around the edges of the scene
    scene_boundary = rectangle(instance_dict["image_tensor"])
    frame_box = extruded_polygon(scene_boundary, d_border)
    scene_boundary_sg = skgeom_rectangle(instance_dict["image_tensor"])
    frame_box_sg = skgeom_extruded_polygon(scene_boundary_sg, d_border)

    no_ego_zones = [frame_box]
    no_ego_zones_sg = [frame_box_sg]
    no_occluder_zones = [frame_box]
    no_occluder_zones_sg = [frame_box_sg]

    # agent_buffers = [shapely_poly_2_skgeom_poly(LineString(np.concatenate((past, future), axis=0)).buffer(r_occluders)) for past, future in zip(instance_dict["pasts"], instance_dict["futures"])]
    agent_buffers = [shapely_poly_2_skgeom_poly(LineString(future).buffer(r_agents)) for future in instance_dict["pasts"]]
    # agent_segments = [trajectory_2_Segment2_list(traj) for traj in instance_dict["pasts"]]

    p_occls = []
    p_disoccls = []
    target_agents_fully_observable_regions = []

    for idx, agent_id in enumerate(instance_dict["agent_ids"]):
        past = instance_dict["pasts"][idx]
        future = instance_dict["futures"][idx]

        if agent_id in target_agents:
            # pick random occlusion and disocclusion points lying on interpolated trajectory
            # p_occl = np.array(random_interpolated_point(past, (min_obs, past.shape[0])))
            # p_disoccl = np.array(random_interpolated_point(future, (0, future.shape[0] - min_reobs)))
            last_obs_timestep = np.random.randint(min_obs - 1, past.shape[0] - 1)
            first_reobs_timestep = np.random.randint(future.shape[0] - min_reobs + 1)

            p_last_obs = past[last_obs_timestep]
            p_first_reobs = future[first_reobs_timestep]

            p_occls.append(p_last_obs)
            p_disoccls.append(p_first_reobs)

            # scene_segments = agent_buffers.copy()
            # scene_segments.pop(idx)
            # scene_segments = [poly.edges for poly in scene_segments]

            # ensure our candidate location for the ego-perceiver is not obstructed by any of the other agents present
            other_buffers = agent_buffers.copy()
            other_buffers.pop(idx)
            # other_segments = agent_segments.copy()
            # other_segments.pop(idx)

            scene_segments = []
            scene_segments.extend(scene_boundary_sg.edges)
            # scene_segments.extend(circ_beg.edges)
            # scene_segments.extend(circ_end.edges)
            [scene_segments.extend(poly.edges) for poly in other_buffers]
            # [scene_segments.extend(segments) for segments in other_segments]

            traj_fully_observable = functools.reduce(
                lambda poly_1, poly_2: sg.boolean_set.intersect(poly_1, poly_2)[0],
                [visibility_polygon(point, scene_segments) for point in np.concatenate([past, future], axis=0)[1:-1]]
            )

            target_agents_fully_observable_regions.append(traj_fully_observable)

            no_ego_zones.extend(target_agent_no_ego_zones(np.concatenate((past, future), axis=0), (d_min_occl_ego + d_min_occl_ag), taper_angle))
            no_occluder_zones.extend(target_agent_no_ego_zones(np.concatenate((past, future), axis=0), d_min_occl_ag, taper_angle))

        else:
            no_ego_zones.append(LineString(np.concatenate((past, future), axis=0)).buffer((d_min_occl_ego + d_min_occl_ag)))
            no_occluder_zones.append(LineString(np.concatenate((past, future), axis=0)).buffer(d_min_occl_ag))


    no_ego_zones = MultiPolygon(no_ego_zones)
    no_occluder_zones = MultiPolygon(no_occluder_zones)

    # extract polygons within which to sample our ego position
    yes_ego_zones = rectangle(instance_dict["image_tensor"]).difference(unary_union(no_ego_zones))
    yes_ego_zones = MultiPolygon([yes_ego_zones]) if isinstance(yes_ego_zones, Polygon) else yes_ego_zones

    # TODO: PROPER INTERSECTION OF YES_EGO_ZONES AND VISIBLE REGION (I.E., GREEN AND BLUE)
    # yes_ego_zones = yes_ego_zones.intersection(skgeom_poly_2_shapely_poly(target_agents_fully_observable_regions[0]))

    yes_triangles = []
    [yes_triangles.extend(polygon_triangulate(zone)) for zone in yes_ego_zones.geoms]
    yes_triangles = MultiPolygon(yes_triangles)

    ego_points = random_points_in_triangles_collection(yes_triangles, k=n_egos)

    # visualization part
    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    # plot agent buffers
    [skgeom_plot_polygon(ax, poly, edgecolor="blue", facecolor="blue", alpha=0.2) for poly in agent_buffers]

    # plot chosen occlusion points
    [ax.scatter(point[0], point[1], marker="x", c="Yellow") for point in p_occls]
    [ax.scatter(point[0], point[1], marker="x", c="Yellow") for point in p_disoccls]

    # plot the fully observable regions
    [skgeom_plot_polygon(ax, poly, facecolor="blue", edgecolor="blue", alpha=0.2) for poly in target_agents_fully_observable_regions]

    # highlight regions generated for the selection of the ego
    [plot_polygon(ax, area, facecolor="red", alpha=0.2) for area in no_ego_zones.geoms]
    [plot_polygon(ax, area, facecolor="green", edgecolor="green", alpha=0.2) for area in yes_triangles.geoms]

    # plot the ego
    ax.scatter(ego_points[:, 0], ego_points[:, 1], marker="x", c="Red")

    plt.show()

    return ego_points


def time_polygon_generation(n_iterations: int = 1000000):
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
    for i in range(n_iterations):
        skgeom_poly_2_shapely_poly(polysg)
    print(f"skgeom 2 shapely polygon conversion: {time() - before}")

    before = time()
    polysp = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    for i in range(n_iterations):
        shapely_poly_2_skgeom_poly(polysp)
    print(f"shapely 2 skgeom polygon conversion: {time() - before}")

if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    ego_points = place_ego(instance_dict=instance_dict)
