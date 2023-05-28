import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import torch

import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import data.sdd_visualize as sdd_visualize
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import matplotlib.collections as mpl_coll
from shapely.geometry import LineString, Polygon, GeometryCollection
from shapely.ops import unary_union, triangulate
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
    voronoi_edges = shapely.ops.voronoi_diagram(polygon, edges=True).intersection(polygon)
    # delaunay triangulation of every point (both from voronoi diagram and the polygon itself)
    candidate_triangles = triangulate(GeometryCollection([voronoi_edges, polygon]))
    # keep only triangles inside original polygon
    return [triangle for triangle in candidate_triangles if triangle.centroid.within(polygon)]


def random_points_in_triangle(triangle: Polygon, k: int = 1) -> np.array:
    # inspired by: https://stackoverflow.com/a/47418580
    x = np.sort(np.random.rand(2, k), axis=0)
    return np.array(triangle.exterior.xy)[:, :-1] @ np.array([x[0], x[1]-x[0], 1.0-x[1]])


def random_points_in_triangles_collection(triangles: List[Polygon], k: int) -> np.array:
    proportions = np.array([tri.area for tri in triangles])
    proportions /= sum(proportions)         # make a vector of probabilities
    points = np.array(
        [random_points_in_triangle(triangles[idx]) for idx in np.random.choice(len(triangles), size=k, p=proportions)]
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


def place_ego(instance_dict: dict):
    agent_id = 12
    idx = instance_dict["agent_ids"].index(agent_id)
    past = instance_dict["pasts"][idx]
    future = instance_dict["futures"][idx]

    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    min_obs = 2     # minimum amount of timesteps we want to have observed within observation window
    min_reobs = 2   # minimum amount of timesteps we want to be able to reobserve after disocclusion


    # pick random occlusion and disocclusion points lying on interpolated trajectory
    p_occl = random_interpolated_point(past, (min_obs, past.shape[0]))
    p_disoccl = random_interpolated_point(future, (0, future.shape[0]-min_reobs))

    # plot occlusion points
    ax.scatter(p_occl[0], p_occl[1], marker="x", c="yellow")
    ax.scatter(p_disoccl[0], p_disoccl[1], marker="x", c="green")

    # generate safety buffer area around agent's trajectory
    r_agents = 60
    target_buffer = LineString(np.concatenate((past, future), axis=0)).buffer(r_agents).convex_hull

    other_buffers = []
    for other in range(len(instance_dict["agent_ids"])):
        if instance_dict["agent_ids"][other] == agent_id:
            continue
        pst = instance_dict["pasts"][other]
        ftr = instance_dict["futures"][other]
        other_buffer = LineString(np.concatenate((pst, ftr), axis=0)).buffer(r_agents)
        other_buffers.append(other_buffer)

    # set safety perimeter around the edges of the scene
    d_border = 120     # pixel distance from scene border
    frame_box = extruded_polygon(rectangle(instance_dict["image_tensor"]), d_border)

    # define no-ego regions based on taper angle, in order to prevent situations where agent is in direction of sight
    taper_angle = 60        # degrees
    u_traj = np.array(future[-1] - past[0])     # unit vector of agent's trajectory (future[-1], past[0])
    u_traj /= np.linalg.norm(u_traj)
    no_ego_1 = bounded_wedge(np.array(future[-1]), -u_traj, float(np.radians(taper_angle)), rectangle(instance_dict["image_tensor"]))
    no_ego_2 = bounded_wedge(np.array(past[0]), u_traj, float(np.radians(taper_angle)), rectangle(instance_dict["image_tensor"]))

    no_egos = [no_ego_1, no_ego_2, target_buffer, frame_box]
    no_egos.extend(other_buffers)

    no_ego = unary_union(no_egos)

    # plot no-ego regions
    plot_polygon(ax, target_buffer, facecolor="red", alpha=0.2)
    plot_polygon(ax, frame_box, facecolor="red", alpha=0.2)
    plot_polygon(ax, no_ego_1, facecolor="red", alpha=0.2)
    plot_polygon(ax, no_ego_2, facecolor="red", alpha=0.2)
    [plot_polygon(ax, other_zone, facecolor="red", alpha=0.2) for other_zone in other_buffers]

    # extract polygons within which to sample our ego position
    yes_ego = [Polygon(hole) for hole in no_ego.interiors]

    print(yes_ego)
    yes_triangles = []
    for zone in yes_ego:
        yes_triangles.extend(polygon_triangulate(zone))

    for poly in yes_triangles:
        plot_polygon(ax, poly, facecolor="green", edgecolor="green", alpha=0.2)

    ego_points = random_points_in_triangles_collection(yes_triangles, k=1)

    ax.scatter(ego_points[:, 0], ego_points[:, 1], marker="x", c="black")

    # TODO
    # WIP: generation of virtual occluding wall

    for row_idx in range(ego_points.shape[0]):

        wall_1 = point_between(ego_points[row_idx], np.array(p_occl), np.random.random())
        wall_2 = point_between(ego_points[row_idx], np.array(p_disoccl), np.random.random())

        ax.plot([wall_1[0], wall_2[0]], [wall_1[1], wall_2[1]], c="black")

    plt.show()


if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    place_ego(instance_dict=instance_dict)


