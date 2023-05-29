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
from shapely.geometry import LineString, Polygon, GeometryCollection, MultiPolygon
from shapely.ops import unary_union, triangulate, voronoi_diagram
from typing import List, Tuple


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


def plot_polygon(ax: matplotlib.axes.Axes, poly: Polygon, **kwargs) -> None:
    path = mpl_path.Path.make_compound_path(
        mpl_path.Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[mpl_path.Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

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


def extruded_polygon(polygon: Polygon, d_border: float) -> Polygon:
    hole = polygon.buffer(-d_border)
    return Polygon(shell=polygon.exterior.coords, holes=[hole.exterior.coords])


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


def place_ego(instance_dict: dict):
    n_targets = 1       # [-]   number of desired target agents to occlude virtually
    d_border = 120      # [px]  distance from scene border
    min_obs = 2         # [-]   minimum amount of timesteps we want to have observed within observation window
    min_reobs = 2       # [-]   minimum amount of timesteps we want to be able to reobserve after disocclusion
    r_agents = 60       # [px]  safety buffer around agents for placement of ego
    taper_angle = 60    # [deg] angle for the generation of wedges
    n_egos = 10          # [-]   number of candidate positions to sample for the simulated ego

    target_agents = select_random_target_agents(instance_dict, n_targets)

    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    # set safety perimeter around the edges of the scene
    frame_box = extruded_polygon(rectangle(instance_dict["image_tensor"]), d_border)

    no_ego_zones = [frame_box]

    for agent_id in instance_dict["agent_ids"]:
        idx = instance_dict["agent_ids"].index(agent_id)
        past = instance_dict["pasts"][idx]
        future = instance_dict["futures"][idx]

        if agent_id in target_agents:
            # pick random occlusion and disocclusion points lying on interpolated trajectory
            p_occl = random_interpolated_point(past, (min_obs, past.shape[0]))
            p_disoccl = random_interpolated_point(future, (0, future.shape[0] - min_reobs))

            # plot occlusion points
            ax.scatter(p_occl[0], p_occl[1], marker="x", c="yellow")
            ax.scatter(p_disoccl[0], p_disoccl[1], marker="x", c="green")

            no_ego_zones.extend(target_agent_no_ego_zones(np.concatenate((past, future), axis=0), r_agents, taper_angle))
        else:
            other_buffer = LineString(np.concatenate((past, future), axis=0)).buffer(r_agents)
            no_ego_zones.append(other_buffer)

    no_ego_zones = MultiPolygon(no_ego_zones)

    # extract polygons within which to sample our ego position

    yes_ego_zones = rectangle(instance_dict["image_tensor"]).difference(unary_union(no_ego_zones))
    yes_ego_zones = MultiPolygon([yes_ego_zones]) if isinstance(yes_ego_zones, Polygon) else yes_ego_zones

    yes_triangles = []
    for zone in yes_ego_zones.geoms:
        yes_triangles.extend(polygon_triangulate(zone))
    yes_triangles = MultiPolygon(yes_triangles)

    # highlight regions generated for the selection of the ego
    [plot_polygon(ax, area, facecolor="red", alpha=0.2) for area in no_ego_zones.geoms]
    [plot_polygon(ax, area, facecolor="green", edgecolor="green", alpha=0.2) for area in yes_triangles.geoms]

    ego_points = random_points_in_triangles_collection(yes_triangles, k=n_egos)

    ax.scatter(ego_points[:, 0], ego_points[:, 1], marker="x")

    for ego_point in ego_points:
        print(ego_point)
        # todo: stuff with the generation of virtual occluders

    plt.show()

    return ego_points


if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    ego_points = place_ego(instance_dict=instance_dict)
