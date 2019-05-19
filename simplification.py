from __future__ import print_function, division

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.optimize import *
from math import *
import cvxopt   
import PIL.Image as Image  
import sys
    
from trimesh import TriMesh

def convex_hull_to_obj(hull, filepath):
    vertices = hull.points[hull.vertices]
    points_index = -1 * np.ones(hull.points.shape[0], dtype = np.int32)
    points_index[hull.vertices] = np.arange(len(hull.vertices))
    faces = np.array([points_index[hface] for hface in hull.simplices]) + 1
    
    for index in range(len(faces)):
        face = vertices[faces[index] - 1]
        normals = hull.equations[index, :3]

        n = np.cross(face[1] - face[0], face[2] - face[0])
        if np.dot(normals, n) < 0:
            faces[index][[1, 0]] = faces[index][[0, 1]]
            
    obj_file = open(filepath, 'w')
    for index in range(vertices.shape[0]):
        obj_file.write('v ' + str(vertices[index][0]) + ' ')
        obj_file.write(str(vertices[index][1]) + ' ')
        obj_file.write(str(vertices[index][2]) + '\n')

    for index in range(faces.shape[0]):
        obj_file.write('f ' + str(faces[index][0]) + ' ' + str(faces[index][1]) + ' ' + str(faces[index][2]) + '\n')
    obj_file.close()
     
def tetrahedron_volume(face, point):
    n = np.cross(face[1] - face[0], face[2] - face[0])
    return abs(np.dot(n, point - face[0])) / 6.0

def contract_edges(mesh):
    edges = mesh.get_edges()
    faces = mesh.faces
    vertices = mesh.vs
    
    temp_list1 = []
    temp_list2 = []
    count = 0

    for edge_index in range(len(edges)):
        edge=edges[edge_index]
        vertex1 = edge[0]
        vertex2 = edge[1]
        related_faces_1 = mesh.vertex_face_neighbors(vertex1)
        related_faces_2 = mesh.vertex_face_neighbors(vertex2)

        related_faces = list(set(related_faces_1) | set(related_faces_2))
        related_faces = [faces[index] for index in related_faces]
        old_face_list = []
        
        c = np.zeros(3)
        A = []
        b = []

        for index in range(len(related_faces)):
            face = related_faces[index]
            p0 = vertices[face[0]]
            p1 = vertices[face[1]]
            p2 = vertices[face[2]]
            old_face_list.append(np.asarray([p0, p1, p2]))
            
            n = np.cross(p1 - p0, p2 - p0)
            
            n = n / np.sqrt(np.dot(n, n))
            
            A.append(n)
            b.append(np.dot(n, p0))
            c += n
                            
        A = -np.asfarray(A)
        b = -np.asfarray(b)
        
        c = np.asfarray(c)
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
        res = cvxopt.solvers.lp(cvxopt.matrix(c), cvxopt.matrix(A), cvxopt.matrix(b), solver = 'glpk')

        if res['status'] == 'optimal':
            newpoint = np.asfarray(res['x']).squeeze()

            tetra_volume_list = []
            for each_face in old_face_list:
                tetra_volume_list.append(tetrahedron_volume(each_face, newpoint))
            volume = np.asarray(tetra_volume_list).sum()

            temp_list1.append((count, volume, vertex1, vertex2))
            temp_list2.append(newpoint)
            count += 1
        
    if len(temp_list1) == 0:
        pass
    else:
        min_tuple=min(temp_list1, key=lambda x: x[1])
        final_index = min_tuple[0]
        final_point = temp_list2[final_index]
        
        v1_ind = min_tuple[2]
        v2_ind = min_tuple[3]

        related_faces_1 = mesh.vertex_face_neighbors(v1_ind)
        related_faces_2 = mesh.vertex_face_neighbors(v2_ind)
        related_faces = list(set(related_faces_1) | set(related_faces_2))

        related_faces_vertex_ind = [faces[index] for index in related_faces]

        if len((set(mesh.vertex_vertex_neighbors(v1_ind)).intersection(set(mesh.vertex_vertex_neighbors(v2_ind))))) != 2:
            print('Impossible to remove edge as link condition violated.')
        
        old2new = mesh.remove_vertex_indices([v1_ind, v2_ind])

        new_vertex_index = current_vertices_num = len(old2new[old2new != -1])

        new_faces_vertex_ind = []
        
        for face in related_faces_vertex_ind:
            new_face = [new_vertex_index if x == v1_ind or x == v2_ind else old2new[x] for x in face]
            if len(set(new_face)) == len(new_face):
                new_faces_vertex_ind.append(new_face)
        
        mesh.vs = np.vstack((mesh.vs, final_point))
        mesh.faces = np.vstack((mesh.faces, new_faces_vertex_ind))
        
        mesh.topology_changed()

    return mesh
        