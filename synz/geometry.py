"""
geometry utils
Author: Xianghui, December 16, 2021
"""
import time

import numpy as np
from psbody.mesh import Mesh

from scipy.spatial.transform import Rotation

class GeometryUtils:
    def __init__(self):
        pass

    @staticmethod
    def simplify_object(obj:Mesh, target_face=8000)->Mesh:
        from open3d.cpu.pybind.geometry import TriangleMesh # this works for 0.12.0
        from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector
        if len(obj.f) <= target_face:
            return obj

        start = time.time()
        verts = Vector3dVector(obj.v)
        faces = Vector3iVector(obj.f)

        mesh_o3d = TriangleMesh(vertices=verts, triangles=faces)
        mesh_simp = mesh_o3d.simplify_quadric_decimation(target_face, 0.5)
        mesh = Mesh(v=np.array(mesh_simp.vertices), f=np.array(mesh_simp.triangles))
        end = time.time()
        print(f"Original: {len(verts)} vertices, {len(faces)} faces, simplified: {len(mesh.v)} vertices, {len(mesh.f)} faces, time: {end-start:.4f}")

        return mesh

    @staticmethod
    def axis_angle_to_rot(angle):
        "convert axis angle to rotation matrix"
        rot = Rotation.from_rotvec(angle)
        return rot.as_matrix()

def test():
    "test axis angle transformation"
    import sys, os
    sys.path.append(os.getcwd())
    from behave.frame_data import FrameDataReader
    from psbody.mesh import Mesh, MeshViewer
    seq = "/BS/xxie-4/work/kindata/Sep29_shuo_chairblack_hand"
    temp_obj = "/BS/xxie2020/work/objects/chair_black/chair_black_f2500.ply"
    obj = Mesh()
    obj.load_from_file(temp_obj)
    obj.v -= np.mean(obj.v, 0)
    reader = FrameDataReader(seq)

    obj_name = 'fit01'
    idx = 10
    angle, trans = reader.get_objfit_params(idx, obj_name)
    obj_fit = reader.get_objfit(idx, obj_name)
    rot = GeometryUtils.axis_angle_to_rot(angle)

    obj.v = np.matmul(obj.v, rot.T) + trans

    mv = MeshViewer()
    mv.set_static_meshes([obj, obj_fit])


if __name__ == '__main__':
    test()