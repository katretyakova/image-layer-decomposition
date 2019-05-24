# -*- coding: utf-8 -*-
from __future__ import print_function, division

import time
import warnings
import json
import time
import PIL.Image as Image
from simplification import *
import scipy.sparse
import scipy.optimize
import scipy
from trimesh import *

import pyximport
pyximport.install(reload_support = True)
from GteDistPointTriangle import *

def rmse(hull_vertices, points, counts):
    hull = ConvexHull(hull_vertices)
    triangulation = Delaunay(hull_vertices)
    ind = triangulation.find_simplex(points, tol = 1e-8)
    min_distances = []
    for i in range(points.shape[0]):
        if ind[i]<0:
            distances = []
            for j in range(hull.simplices.shape[0]):
                result = DCPPointTriangle(points[i], hull.points[hull.simplices[j]])
                distances.append(result['distance'])
            min_distances.append(min(distances))
    min_distances = np.asarray(min_distances)
    
    return (((min_distances ** 2) * counts[ind < 0]).sum() / counts.sum()) ** 0.5
                
from collections import Counter

def get_unique_colors_and_their_counts(arr):
    unique_colors, counts = np.unique(arr, axis = 0, return_counts = True)
    return unique_colors, counts
                
def simplify(data, output_prefix, num_thres = 0.1, error_threshold = 2.0 / 255.0):
    rgb_palette_hull = ConvexHull(data.reshape((-1, 3)))
    origin_vertices = rgb_palette_hull.points[ rgb_palette_hull.vertices ]

    output_rawhull_obj_file = output_prefix + '.obj'
    convex_hull_to_obj(rgb_palette_hull, output_rawhull_obj_file)    
    
    new_data = (((data * 255).round().astype(np.uint8)//8)*8+4)/255.0
    unique_data, pixel_counts = get_unique_colors_and_their_counts(new_data.reshape((-1,3)))

    max_loop = 5000
    for i in range(max_loop):
        if i % 10 == 0:
            print ("loop: ", i)
        mesh = TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
        old_num = len(mesh.vs)
        old_vertices = mesh.vs
        mesh = contract_edges(mesh)
        rgb_palette_hull = ConvexHull(mesh.vs)
        convex_hull_to_obj(rgb_palette_hull, output_rawhull_obj_file)

        if len(rgb_palette_hull.vertices) <= 10:
            RMSE = rmse(rgb_palette_hull.points[rgb_palette_hull.vertices].clip(0.0, 1.0), unique_data, pixel_counts)
                
            if RMSE > error_threshold:
                oldhull = ConvexHull(old_vertices)
                return oldhull.points[oldhull.vertices].clip(0.0, 1.0)
    
        if len(rgb_palette_hull.vertices) == old_num or len(rgb_palette_hull.vertices) == 4:
            return rgb_palette_hull.points[rgb_palette_hull.vertices].clip(0.0, 1.0)
        
def find_barycentric_coordinates(tetrahedron, points):
    triangulation = Delaunay(tetrahedron)    
    simplex = triangulation.find_simplex(points, tol = 1e-6)

    X = triangulation.transform[simplex, :points.shape[1]]
    Y = points - triangulation.transform[simplex, points.shape[1]]

    b = np.einsum('...jk,...k->...j', X, Y)
    barycentric_coordinates = np.c_[b, 1 - b.sum(axis = 1)]

    return triangulation, barycentric_coordinates, simplex

def rgbxy_weights(tetrahedron, points):
    triangulation, barycentric_coordinates, simplex = find_barycentric_coordinates(tetrahedron, points)

    rows = np.repeat(np.arange(len(points)).reshape((-1, 1)), len(triangulation.simplices[0]), 1).ravel()
        
    cols = triangulation.simplices[simplex].ravel()
    vals = barycentric_coordinates.ravel()
    weights = scipy.sparse.coo_matrix((vals, (rows, cols)), shape = (len(points), len(tetrahedron))).tocsr()

    return weights

def palette_weights(tetrahedron, points):
    triangulation, barycentric_coordinates, simplex = find_barycentric_coordinates(tetrahedron, points)

    weights = np.zeros((points.shape[0], tetrahedron.shape[0]))
    point_indices = np.arange(len(points))

    for i in range(len(triangulation.simplices)):
        weights[point_indices[simplex == i][:, None], np.array(triangulation.simplices[i])] = barycentric_coordinates[point_indices[simplex == i], :]

    return weights    

def get_layer_order(rgb_palette):
    diff = abs(rgb_palette - np.array([[0, 0, 0]])).sum(axis = -1)
    order = np.argsort(diff)

    return order

def project_outside_points(triangulation, hull_vertices, rgb_palette_hull):
    relevant_simplices_indices = triangulation.find_simplex(hull_vertices, tol = 1e-8)

    for i in range(hull_vertices.shape[0]):
        if relevant_simplices_indices[i] == -1:
            distances = []
            closest_points = []
            for j in range(rgb_palette_hull.simplices.shape[0]):
                result = DCPPointTriangle(hull_vertices[i], rgb_palette_hull.points[rgb_palette_hull.simplices[j]])
                distances.append(result['distance'])
                closest_points.append(result['closest'])
            distances = np.asarray(distances)
            index = np.argmin(distances)
            hull_vertices[i] = closest_points[index]

    return hull_vertices

def assign_points_to_palette_faces(rgb_palette_hull, ordered_rgb_palette, unique_colors):
    tetra_pixel_dict = {}
    for face_vertex_ind in rgb_palette_hull.simplices:
        if (face_vertex_ind != 0).all():
            i, j, k = face_vertex_ind
            tetra_pixel_dict.setdefault(tuple((i, j, k)), [])

    index_list = np.array(list(np.arange(len(unique_colors))))

    for face_vertex_ind in rgb_palette_hull.simplices:
        if (face_vertex_ind != 0).all():
            i, j, k = face_vertex_ind
            tetra = np.array([ordered_rgb_palette[0], ordered_rgb_palette[i], ordered_rgb_palette[j], ordered_rgb_palette[k]])
            try:
                test_Del = Delaunay(tetra)
                if len(index_list) != 0:
                    relevant_simplices_indices = test_Del.find_simplex(unique_colors[index_list], tol = 1e-8)
                    chosen_index = list(index_list[relevant_simplices_indices >= 0])
                    tetra_pixel_dict[tuple((i, j, k))] += chosen_index
                    index_list = np.array(list(set(index_list) - set(chosen_index)))
            except Exception as e:
                pass

    return tetra_pixel_dict, index_list

def get_unique_weights(unique_colors, ordered_rgb_palette, tetra_pixel_dict):
    shortest_path_order = tuple(np.arange(len(ordered_rgb_palette))[1:])
    unique_weights = np.zeros((unique_colors.shape[0], len(ordered_rgb_palette)))

    for vertice_tuple in tetra_pixel_dict:
        ordered_vertices = np.asarray(shortest_path_order)[np.asarray(sorted(list(shortest_path_order).index(s) for s in vertice_tuple))]
        ordered_vertex_tuple = tuple(list(ordered_vertices))
        
        colors = np.array([ordered_rgb_palette[0],
                         ordered_rgb_palette[ordered_vertex_tuple[0]],
                         ordered_rgb_palette[ordered_vertex_tuple[1]],
                         ordered_rgb_palette[ordered_vertex_tuple[2]]
                        ])
        pixel_indices = np.array(tetra_pixel_dict[vertice_tuple])
        if len(pixel_indices) != 0:
            arr = unique_colors[pixel_indices]
            Y = palette_weights(colors, arr)
            unique_weights[pixel_indices[:, None], np.array([0] + list(ordered_vertex_tuple))] = Y.reshape((arr.shape[0], -1))
    
    return unique_weights

def rgb_weights(original_hull_vertices, rgb_palette):
    hull_vertices = original_hull_vertices.copy()

    order = get_layer_order(rgb_palette)

    img_shape = hull_vertices.shape
    hull_vertices = hull_vertices.reshape((-1, 3))

    ordered_rgb_palette = rgb_palette[order]

    rgb_palette_hull = ConvexHull(ordered_rgb_palette)
    triangulation = Delaunay(ordered_rgb_palette)

    hull_vertices = project_outside_points(triangulation, hull_vertices, rgb_palette_hull)

    colors2xy = {}
    unique_hull_vertices = list(set(list(tuple(element) for element in hull_vertices)))

    for element in unique_hull_vertices:
        colors2xy.setdefault(tuple(element), [])
        
    for index in range(len(hull_vertices)):
        element = hull_vertices[index]
        colors2xy[tuple(element)].append(index)

    unique_colors = np.array(list(colors2xy.keys()))

    tetra_pixel_dict, index_list = assign_points_to_palette_faces(rgb_palette_hull, ordered_rgb_palette, unique_colors)
    unique_weights = get_unique_weights(unique_colors, ordered_rgb_palette, tetra_pixel_dict)

    mixing_weights = np.zeros((len(hull_vertices), len(ordered_rgb_palette)))
    for index in range(len(unique_colors)):
        element = unique_colors[index]
        index_list = colors2xy[tuple(element)]
        mixing_weights[index_list, :] = unique_weights[index, :]
    
    ordered_mixing_weights = np.ones(mixing_weights.shape)
    ordered_mixing_weights[:, order] = mixing_weights

    return ordered_mixing_weights.reshape((img_shape[0], img_shape[1], -1))

