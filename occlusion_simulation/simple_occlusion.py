import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import data.sdd_dataloader as sdd_dataloader
import data.sdd_extract as sdd_extract
import data.sdd_visualize as sdd_visualize
import matplotlib.path as mpl_path
import matplotlib.patches as mpl_patches
import matplotlib.collections as mpl_coll
from shapely.geometry import LineString, Polygon, Point, GeometryCollection
from shapely.ops import unary_union, triangulate
from shapely.affinity import affine_transform


def point_between(point_1: np.array, point_2: np.array, k):
    # k between 0 and 1, point_1 and point_2 of same shape
    return k * point_1 + (1 - k) * point_2


def rotation_matrix(theta):
    # angle in radians
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def no_ego_cone(point, u, theta, frame_box):
    # generate the polygon corresponding to the intersection of:
    # an infinite cone pointing towards unit_vec u and with a taper angle 2*theta, and with extremety located at point
    # and the frame_box polygon
    big_vec = -u * 2 * np.max(frame_box.bounds)        # large vector, to guarantee cone is equivalent to infinite

    p1 = rotation_matrix(theta) @ big_vec + point
    p2 = rotation_matrix(-theta) @ big_vec + point

    p3 = p1 + big_vec
    p4 = p2 + big_vec

    return Polygon([point, p1, p3, p4, p2, point]).intersection(frame_box)


# def plot_polygon(ax, poly):
#     fill_patch = mpl_patches.Polygon(list(zip(*poly.exterior.xy)), facecolor='red', alpha=0.2)
#     ax.plot(*poly.exterior.xy, c="red", alpha=0.1)
#     ax.add_patch(fill_patch)


def plot_polygon(ax, poly, **kwargs):
    path = mpl_path.Path.make_compound_path(
        mpl_path.Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[mpl_path.Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = mpl_patches.PathPatch(path, **kwargs)
    collection = mpl_coll.PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    return collection


def polygon_triangulate(polygon):
    # this approach solves the triangulation of non-convex polygons (with eventual interiors)
    # create voronoi edges of polygon (bounded within the polygon itself)
    voronoi_edges = shapely.ops.voronoi_diagram(polygon, edges=True).intersection(polygon)
    # delaunay triangulation of every point (both from voronoi diagram and the polygon itself)
    candidate_triangles = triangulate(GeometryCollection([voronoi_edges, polygon]))
    # keep only triangles inside original polygon
    return [triangle for triangle in candidate_triangles if triangle.centroid.within(polygon)]


def random_point_in_triangle(triangle):
    x = np.random.rand(2)
    print(triangle)


def random_points_in_polygon(polygon, k):
    # taken from :
    # https://codereview.stackexchange.com/a/204289

    # TODO: https://stackoverflow.com/a/47418580

    # WIP WIP WIP DEFINITELY NOT READY
    triangles = polygon_triangulate(polygon)
    areas = [tri.area for tri in triangles]

    print(areas)
    # print(transforms)
    # print(len(transforms))
    points = []

    for rand_idx in np.random.choice(len(areas), size=k, p=[float(i) / sum(areas) for i in areas]):
        print(rand_idx)
        # do something here
        print(random_point_in_triangle(polygon_triangulate(triangles[rand_idx])))
    return points


def place_ego(instance_dict: dict):

    agent_id = 12
    idx = instance_dict["agent_ids"].index(agent_id)
    past = instance_dict["pasts"][idx]
    future = instance_dict["futures"][idx]
    label = instance_dict["labels"][idx]
    full_obs = instance_dict["is_fully_observed"][idx]

    fig, ax = plt.subplots()
    sdd_visualize.visualize_training_instance(ax, instance_dict=instance_dict)

    # pick random past point
    min_obs = 2     # minimum amount of timesteps we want to have observed within observation window
    T_occl = np.random.randint(0, past.shape[0] - min_obs) # number of timesteps to move to the past from t_0 (i.e., 0 -> t_0, 1 -> t_-1, etc)
    past_idx = -1-T_occl

    # pick random future point
    min_reobs = 2   # minimum amount of timesteps we want to be able to reobserve after disocclusion
    T_disoccl = future_idx = np.random.randint(0, future.shape[0] - min_reobs)

    # pick random points for start and end of occlusion region
    p_occl = point_between(past[past_idx], past[past_idx-1], np.random.random())
    p_disoccl = point_between(future[future_idx], future[future_idx+1], np.random.random())

    # find midpoint
    midpoint = point_between(p_occl, p_disoccl, 0.5)

    # plot occlusion points
    ax.scatter(past[past_idx][0], past[past_idx][1], marker="x", c="red")
    ax.scatter(future[future_idx][0], future[future_idx][1], marker="x", c="orange")
    ax.scatter(p_occl[0], p_occl[1], marker="x", c="yellow")
    ax.scatter(p_disoccl[0], p_disoccl[1], marker="x", c="green")
    ax.scatter(midpoint[0], midpoint[1], marker="x", c="blue")

    # generate safety buffer area around agent's trajectory
    r_agents = 60
    too_close = LineString(np.concatenate((past, future), axis=0)).buffer(r_agents).convex_hull

    # plot safety buffer
    # plot_polygon(ax, too_close)

    # set safety perimeter around the edges of the scene
    d_border = 120     # pixel distance from scene border
    y_img, x_img = instance_dict["image_tensor"].shape[1:]

    frame_box = Polygon([(0, 0),
                         (x_img, 0),
                         (x_img, y_img),
                         (0, y_img),
                         (0, 0)])

    border_box = frame_box.buffer(-d_border)
    no_ego_border = Polygon(shell=frame_box, holes=border_box)

    # plot border box
    # border_box_x, border_box_y = border_box.exterior.xy
    # ax.plot(border_box_x, border_box_y, color="black", alpha=0.3)

    # define no-ego regions based on taper angle, in order to prevent situations where agent is in direction of sight
    taper_angle = 60        # degrees
    u_traj = np.array(future[-1] - past[0])     # unit vector of agent's trajectory (future[-1], past[0])
    u_traj = u_traj / np.linalg.norm(u_traj)
    no_ego_1 = no_ego_cone(np.array(future[-1]), -u_traj, np.radians(taper_angle), frame_box)
    no_ego_2 = no_ego_cone(np.array(past[0]), u_traj, np.radians(taper_angle), frame_box)

    # plot no-ego regions
    # plot_polygon(ax, no_ego_1)
    # plot_polygon(ax, no_ego_2)

    nu = Polygon(frame_box.exterior.coords, [border_box.exterior.coords])
    no_ego = unary_union((no_ego_1, no_ego_2, too_close, nu))

    # plot_polygon(ax, no_ego, facecolor="red", edgecolor="red", alpha=0.2)

    # extract polygons within which to sample our ego position
    yes_ego = [Polygon(hole) for hole in no_ego.interiors]

    zone = yes_ego[0]
    # add a hole to check.
    # zone = Polygon(zone.exterior.coords, [Polygon([(1000, 250), (1300, 250), (1200, 400), (1000, 250)]).exterior.coords])
    # plot_polygon(ax, zone, facecolor="blue", edgecolor="blue", alpha=0.2)

    triangles = polygon_triangulate(zone)

    print(triangles)
    for poly in triangles:
        plot_polygon(ax, poly, facecolor="green", edgecolor="green", alpha=0.2)

    for zone in yes_ego:
        plot_polygon(ax, zone, facecolor="blue", edgecolor="blue", alpha=0.2)

        # print(random_points_in_polygon(zone, 3))
        ego_points = random_points_in_polygon(zone, 100)

        xs = [point.x for point in ego_points]
        ys = [point.y for point in ego_points]
        ax.scatter(xs, ys)

    plt.show()

    print("HEY HEY HEY \n\n\n")
    hey_square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    hey_points = random_points_in_polygon(hey_square, 3)

    fig, ax = plt.subplots()

    for i in polygon_triangulate(hey_square):
        plot_polygon(ax, i, edgecolor="red")

    hyx = [point.x for point in hey_points]
    hyy = [point.y for point in hey_points]
    ax.scatter(hyx, hyy, color="red")

    plt.show()



if __name__ == '__main__':
    print("Ok, let's do this")

    instance_idx = 36225

    config = sdd_extract.get_config()
    dataset = sdd_dataloader.StanfordDroneDataset(config_dict=config)

    instance_dict = dataset.__getitem__(instance_idx)

    place_ego(instance_dict=instance_dict)


