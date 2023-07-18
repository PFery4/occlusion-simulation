import numpy as np
import skgeom as sg
import shapely.geometry as sp
import shapely.ops as spops
import functools
import itertools
from typing import List
import src.occlusion_simulation.type_conversion as type_conv


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
    triangles = [type_conv.skgeom_poly_2_shapely_poly(sg.PolygonWithHoles(tri, [])) for tri in triangles]
    points = np.array(
        [random_points_in_triangle(triangles[idx]) for idx in
         np.random.choice(len(triangles), size=k, p=proportions)]
    ).reshape((k, 2))
    return points


def sample_triangles(triangles: List[sg.Polygon], k: int = 1) -> List[sg.Polygon]:
    proportions = np.array([float(tri.area()) for tri in triangles])
    proportions /= sum(proportions)         # make a vector of probabilities
    return [triangles[idx] for idx in np.random.choice(len(triangles), size=k, p=proportions)]


def skgeom_extruded_polygon(polygon: sg.Polygon, d_border: float) -> sg.PolygonWithHoles:
    skel = sg.skeleton.create_interior_straight_skeleton(polygon)
    return functools.reduce(lambda a, b: sg.boolean_set.difference(a, b)[0], skel.offset_polygons(d_border), polygon)


def triangulate_polyset(polyset: sg.PolygonSet) -> List[sg.Polygon]:
    triangles = list(itertools.chain(*[
        polygon_triangulate(type_conv.skgeom_poly_2_shapely_poly(poly)) for poly in polyset.polygons
    ]))
    return [type_conv.shapely_poly_2_skgeom_poly(triangle) for triangle in triangles]
