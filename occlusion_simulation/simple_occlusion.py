import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
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


def random_points_in_triangle(triangle: Polygon, k: int = 1) -> np.array:
    # inspired by: https://stackoverflow.com/a/47418580
    x = np.sort(np.random.rand(2, k), axis=0)
    return np.array(triangle.exterior.xy)[:, :-1] @ np.array([x[0], x[1]-x[0], 1.0-x[1]])


def random_points_in_triangles_collection(triangles: MultiPolygon, k: int) -> np.array:
    proportions = np.array([tri.area for tri in triangles.geoms])
    proportions /= sum(proportions)         # make a vector of probabilities
    points = np.array(
        [random_points_in_triangle(triangles.geoms[idx]) for idx in
         np.random.choice(len(triangles.geoms), size=k, p=proportions)]
    ).reshape((k, 2))
    return points


# def random_interpolated_point(traj_seq: np.array, timestep_bounds: Tuple[int, int]):
#     """
#     sample a random point among the interpolated sequence of coordinates traj_seq. the point will lie within the
#     interval specified by timestep_bounds (both inclusive)
#     :param traj_seq: shape [Timesteps, Dimensions]
#     :param timestep_bounds: tuple (begin, end)
#     """
#     t_idx = np.random.randint(timestep_bounds[0], timestep_bounds[1]-1)
#     return point_between(traj_seq[t_idx], traj_seq[t_idx+1], np.random.random())


# def rectangle(image_tensor: torch.Tensor) -> Polygon:
#     # returns a rectangle with dimensions corresponding to those of the input image_tensor
#     y_img, x_img = image_tensor.shape[1:]
#     return Polygon([(0, 0), (x_img, 0), (x_img, y_img), (0, y_img), (0, 0)])


# def skgeom_rectangle(image_tensor: torch.Tensor):
#     y_img, x_img = image_tensor.shape[1:]
#     return sg.Polygon([[0, 0], [x_img, 0], [x_img, y_img], [0, y_img]])


def default_rectangle(corner_coords: Tuple[float, float]) -> sg.Polygon:
    """
    WARNING: having any of the two input values equal to 0.0 can result in errors
    :param corner_coords: tuple [height, width], as can be extracted from an image torch.Tensor of shape [C, H, W]
    """
    y, x = corner_coords
    return sg.Polygon([[0, 0], [x, 0], [x, y], [0, y]])


# def extruded_polygon(polygon: Polygon, d_border: float) -> Polygon:
#     hole = polygon.buffer(-d_border)
#     return Polygon(shell=polygon.exterior.coords, holes=[hole.exterior.coords])


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


def target_agent_no_ego_zones(boundary: sg.Polygon, traj: np.array, radius: float = 60, wedge_angle: float = 60) -> List[sg.Polygon]:
    # generate safety buffer area around agent's trajectory
    target_buffer = shapely_poly_2_skgeom_poly(LineString(traj).buffer(radius).convex_hull)

    # define no-ego wedges based on taper angle, in order to prevent situations where ego
    # is directly aligned with target agent's trajectory
    u_traj = np.array(traj[-1] - traj[0])  # unit vector of agent's trajectory (future[-1], past[0])
    u_traj /= np.linalg.norm(u_traj)
    no_ego_1 = bounded_wedge(
        p=(np.array(traj[-1]) - u_traj * radius),
        u=-u_traj,
        theta=float(np.radians(wedge_angle)),
        boundary=boundary
    )
    no_ego_2 = bounded_wedge(
        p=(np.array(traj[0]) + u_traj * radius),
        u=u_traj,
        theta=float(np.radians(wedge_angle)),
        boundary=boundary
    )
    return [target_buffer, no_ego_1, no_ego_2]


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


def place_ego(instance_dict: dict):
    n_targets = 1           # [-]   number of desired target agents to occlude virtually
    n_egos = 1              # [-]   number of candidate positions to sample for the simulated ego

    min_obs = 4             # [-]   minimum amount of timesteps we want to have observed within observation window
    min_reobs = 2           # [-]   minimum amount of timesteps we want to be able to reobserve after disocclusion

    d_border = 200          # [px]  distance from scene border
    d_min_occl_ag = 60      # [px]  minimum distance that any point of a virtual occluder may have wrt any agent
    d_min_occl_ego = 30     # [px]  minimum distance that any point of a virtual occluder may have wrt ego
    taper_angle = 45        # [deg] angle for the generation of wedges

    r_agents = 10           # [px]  how "wide" we approximate agents to be

    target_agents = select_random_target_agents(instance_dict, n_targets)

    # set safety perimeter around the edges of the scene
    scene_boundary = default_rectangle(instance_dict["image_tensor"].shape[1:])
    frame_box = skgeom_extruded_polygon(scene_boundary, d_border=d_border)

    # define no_ego_zones, a list of polygons, within which we wish not to place the ego
    no_ego_zones = []
    # define no_occluder_zones, a list of polygons, within which we wish not to place any virtual occluder
    no_occluder_zones = []

    # define agent_buffers, a list of polygons
    # corresponding to the past trajectories of every agent, inflated by some small radius
    agent_buffers = [
        shapely_poly_2_skgeom_poly(LineString(future).buffer(r_agents)) for future in instance_dict["pasts"]
    ]       # todo: eventually find a way to do this without shapely?

    # lists to keep track of target agents' desired occlusion timesteps
    # (first and last timesteps surrounding the occlusion)
    t_occls = []
    t_disoccls = []

    # list: a given item is a sg.PolygonSet, which describes the regions in space from which
    # every timestep of that agent can be directly observed, unobstructed by other agents
    # (specifically, by their agent_buffer)
    target_agents_fully_observable_regions = []

    for idx, agent_id in enumerate(instance_dict["agent_ids"]):
        past_traj = instance_dict["pasts"][idx]
        future_traj = instance_dict["futures"][idx]
        full_traj = np.concatenate((past_traj, future_traj), axis=0)

        if agent_id in target_agents:
            # pick random occlusion and disocclusion timesteps
            last_obs_timestep = np.random.randint(min_obs - 1, past_traj.shape[0] - 1)
            first_reobs_timestep = np.random.randint(future_traj.shape[0] - min_reobs + 1)
            t_occls.append(last_obs_timestep)
            t_disoccls.append(first_reobs_timestep)

            # to generate the regions within which every coordinate of the target agent is visible, we first need
            # the buffers of every *other* agent
            other_buffers = agent_buffers.copy()
            other_buffers.pop(idx)

            # creating the sg.arrangement.Arragement object necessary to compute the visibility polygon
            scene_segments = list(scene_boundary.edges)
            [scene_segments.extend(poly.edges) for poly in other_buffers]
            scene_arr = sg.arrangement.Arrangement()
            [scene_arr.insert(seg) for seg in scene_segments]

            traj_fully_observable = functools.reduce(
                lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
                [sg.PolygonSet(visibility_polygon(point, arrangement=scene_arr)) for point in full_traj]
            )

            target_agents_fully_observable_regions.append(traj_fully_observable)

            no_ego_zones.extend(
                target_agent_no_ego_zones(
                    boundary=scene_boundary,
                    traj=full_traj,
                    radius=(d_min_occl_ego + d_min_occl_ag) * 1.1,
                    wedge_angle=taper_angle
                )
            )
            no_occluder_zones.append(
                shapely_poly_2_skgeom_poly(
                    LineString(full_traj).buffer(d_min_occl_ag).convex_hull
                )
            )

        else:
            no_ego_zones.append(
                shapely_poly_2_skgeom_poly(
                    LineString(full_traj).buffer((d_min_occl_ego + d_min_occl_ag) * 1.1)
                )
            )
            no_occluder_zones.append(
                shapely_poly_2_skgeom_poly(
                    LineString(full_traj).buffer(d_min_occl_ag)
                )
            )

    no_ego_zones = sg.PolygonSet(no_ego_zones)

    target_agents_fully_observable_regions = functools.reduce(
        lambda polyset_a, polyset_b: polyset_a.intersection(polyset_b),
        target_agents_fully_observable_regions
    )

    # the regions within which we sample our ego are the regions within which target agents' full trajectories are
    # observavle, minus the boundaries and no_ego_zones we set previously
    yes_ego_zones = target_agents_fully_observable_regions.difference(no_ego_zones).difference(frame_box)

    # # extract polygons within which to sample our ego position
    # yes_ego_zones = rectangle(instance_dict["image_tensor"]).difference(unary_union(no_ego_zones))
    # yes_ego_zones = yes_ego_zones.intersection(skgeom_poly_2_shapely_poly(target_agents_fully_observable_regions[0]))
    # yes_ego_zones = MultiPolygon([yes_ego_zones]) if isinstance(yes_ego_zones, Polygon) else yes_ego_zones


    # to sample from yes_ego_zones, we will need to triangulate the regions in yes_ego_zones
    # this can't be done in scikit-geometry, so we're doing it with shapely instead -> todo: maybe possible? try later
    yes_triangles = []
    [yes_triangles.extend(polygon_triangulate(skgeom_poly_2_shapely_poly(zone))) for zone in yes_ego_zones.polygons]
    yes_triangles = MultiPolygon(yes_triangles)     # todo: this should not be a multipolygon, but a sg.PolygonSet

    # TODO: CONTINUE CODE CLEANUP FROM HERE
    # ego_points = random_points_in_triangles_collection(yes_triangles, k=n_egos)
    #
    # TODO: move this down once done with generation of walls
    # visualization part
    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)
    #
    #
    #
    # # COMPUTE OCCLUDERS
    # for ego_point in ego_points:
    #
    #     # draw circle around the ego_point
    #     ego_buffer = Point(*ego_point).buffer(d_min_occl_ego)
    #
    #     no_occluder_zones.append(ego_buffer)
    #
    #     # ITERATE OVER TARGET AGENTS
    #     for agent_id, t_occl, t_disoccl in zip(target_agents, t_occls, t_disoccls):
    #         past_traj = instance_dict["pasts"][instance_dict["agent_ids"].index(agent_id)]
    #         future_traj = instance_dict["futures"][instance_dict["agent_ids"].index(agent_id)]
    #
    #         # triangle defined by ego, and the trajectory segment [t_occl: t_occl+1] of the target agent
    #         occluder_region = Polygon([ego_point, past_traj[t_occl], past_traj[t_occl + 1]])
    #
    #         # extrude no_occluder_regions from th triangle
    #         for no_occl_zone in no_occluder_zones:
    #             occluder_region = occluder_region.difference(no_occl_zone)
    #         occluder_region = MultiPolygon([occluder_region]) if isinstance(occluder_region, Polygon) else occluder_region
    #
    #         # triangulate the resulting region
    #         occluder_triangles = []
    #         [occluder_triangles.extend(polygon_triangulate(poly)) for poly in occluder_region.geoms]
    #         occluder_triangles = MultiPolygon(occluder_triangles)
    #
    #         # sample our first occluder wall coordinate from the region
    #         [p_wall_1] = random_points_in_triangles_collection(occluder_triangles, k=1)
    #
    #         # compute the visibility polygon of this point (corresponds to the regions in space that can be linked to
    #         # the point with a straight line
    #         no_occluder_zones_sg = [shapely_poly_2_skgeom_poly(zone) for zone in no_occluder_zones]
    #         segs = []
    #         segs.extend(scene_boundary_sg.edges)
    #         [segs.extend(poly.edges) for poly in no_occluder_zones_sg]
    #         access_poly = visibility_polygon(p_wall_1, segments=segs)
    #         access_poly = skgeom_poly_2_shapely_poly(access_poly)
    #
    #         other_region = Polygon([ego_point, future_traj[t_disoccl], future_traj[t_disoccl - 1]])     # TODO: PROBLEM WHEN t_disoccl == 0
    #         other_region = other_region.intersection(access_poly)
    #         other_region = MultiPolygon([other_region]) if isinstance(other_region, Polygon) else other_region
    #
    #         other_triangles = []
    #         [other_triangles.extend(polygon_triangulate(poly)) for poly in other_region.geoms]
    #         other_triangles = MultiPolygon(other_triangles)
    #
    #         [p_wall_2] = random_points_in_triangles_collection(other_triangles, k=1)
    #
    #         plot_polygon(ax, access_poly, facecolor="yellow", edgecolor="yellow", alpha=0.2)
    #         ax.scatter(*p_wall_1, marker="x", c="Purple")
    #         ax.scatter(*p_wall_2, marker="x", c="Purple")
    #         ax.plot([p_wall_1[0], p_wall_2[0]], [p_wall_1[1], p_wall_2[1]], c="Purple")
    #
    #         [plot_polygon(ax, zone, facecolor="pink", alpha=0.4) for zone in occluder_region.geoms]
    #         [plot_polygon(ax, zone, facecolor="pink", alpha=0.4) for zone in other_region.geoms]
    #
    # no_occluder_zones = MultiPolygon(no_occluder_zones)




    # plot agent buffers
    [skgeom_plot_polygon(ax, poly, edgecolor="blue", facecolor="blue", alpha=0.2) for poly in agent_buffers]

    # plot chosen occlusion points
    [ax.scatter(
        *instance_dict["pasts"][instance_dict["agent_ids"].index(agent_id)][t_occl], marker="x", c="Yellow"
    ) for agent_id, t_occl in zip(target_agents, t_occls)]
    [ax.scatter(
        *instance_dict["futures"][instance_dict["agent_ids"].index(agent_id)][t_disoccl], marker="x", c="Yellow"
    ) for agent_id, t_disoccl in zip(target_agents, t_disoccls)]

    # plot the fully observable regions
    [skgeom_plot_polygon(ax, poly, facecolor="blue", edgecolor="blue", alpha=0.2) for poly in target_agents_fully_observable_regions.polygons]

    # highlight regions generated for the selection of the ego
    [skgeom_plot_polygon(ax, poly, facecolor="red", alpha=0.2) for poly in no_ego_zones.polygons]
    # [skgeom_plot_polygon(ax, poly, facecolor="green", alpha=0.2) for poly in yes_ego_zones.polygons]
    # [plot_polygon(ax, area, facecolor="orange", alpha=0.2) for area in no_occluder_zones.geoms]
    [plot_polygon(ax, area, facecolor="green", edgecolor="green", alpha=0.2) for area in yes_triangles.geoms]
    #
    # # plot the ego
    # ax.scatter(ego_points[:, 0], ego_points[:, 1], marker="x", c="Red")

    plt.show()

    return ego_points


def time_polygon_generation(instance_dict: dict, n_iterations: int = 1000000):
    print(f"Checking polygon generation timing: {n_iterations} iterations\n")
    # before = time()
    # for i in range(n_iterations):
    #     sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 1), sg.Point2(1, 1), sg.Point2(1, 0)])
    # print(f"skgeom polygon instantiation: {time() - before}")
    #
    # before = time()
    # for i in range(n_iterations):
    #     Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    # print(f"shapely polygon instantiation: {time() - before}")
    #
    # before = time()
    # polysg = sg.Polygon([sg.Point2(0, 0), sg.Point2(0, 1), sg.Point2(1, 1), sg.Point2(1, 0)])
    # polysg = sg.PolygonWithHoles(polysg, [])
    # for i in range(n_iterations):
    #     skgeom_poly_2_shapely_poly(polysg)
    # print(f"skgeom 2 shapely polygon conversion: {time() - before}")
    #
    # before = time()
    # polysp = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
    # for i in range(n_iterations):
    #     shapely_poly_2_skgeom_poly(polysp)
    # print(f"shapely 2 skgeom polygon conversion: {time() - before}")
    #
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


if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    # time_polygon_generation(instance_dict=instance_dict, n_iterations=100000)

    ego_points = place_ego(instance_dict=instance_dict)
