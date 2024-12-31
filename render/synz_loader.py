import sys, os

sys.path.append(os.getcwd())
import pickle as pkl
import numpy as np
from glob import glob
import os.path as osp
import json
from scipy.spatial.transform import Rotation
from copy import deepcopy

from lib_mesh.mesh import Mesh
from lib_smpl.smpl_generator import SMPLHGenerator
from lib_smpl.body_landmark import BodyLandmarks
import paths as paths


class SynzResultLoader:
    "load parameters from the results of synz/synz_batch.py, with possibly some augmentations"
    def __init__(self, params_folder, newshape_root, procigen_asset_root, source, verbose=False):
        """

        :param params_folder: path to optimized H+O parameters, output from synz_batch
        :type params_folder: str
        :param newshape_root: root path to the meshes of new shapes
        :type newshape_root: str
        :param procigen_asset_root: root path to some asset files
        :param source: object shape dataset name
        """
        param_files = self.glob_param_files(params_folder)
        self.param_files = param_files
        self.index = 0  # count which to synthesize

        self.newshape_root = newshape_root
        if len(self.param_files) == 0:
            raise ValueError("No parameter files found!")
        if verbose:
            print(f"In total {len(self.param_files)} parameters found.")

        self.landmark = BodyLandmarks(paths.PROCIGEN_ASSET_ROOT)
        self.scan_geners = json.load(open(f'{procigen_asset_root}/mgn-scan-gender.json'))
        self.verbose = verbose
        self.source = source

    def load(self, smpld_file):
        """
        load HO parameters and return meshes
        :param smpld_file: path to MGN smpld parameter file, e.g. th_good_1/125611487366942/125611487366942_unposed.pkl
        :return:
        :rtype:
        """
        if self.index >= len(self):
            print("All files are iterated, exiting...")
            raise ValueError()

        # load human + object parameters
        ho_params = pkl.load(open(self.param_files[self.index], 'rb'))
        smpld_params = self.load_smpld_params(smpld_file)

        gender = ho_params['gender']  # TODO: load different params if gender does not match
        scan_name = osp.basename(osp.dirname(smpld_file))
        gender_scan = self.scan_geners[scan_name]
        if gender_scan != gender:
            # try to load parameters optimized for another gender
            param_file_new = self.param_files[self.index].replace('.pkl', '_other_gender.pkl')
            if osp.isfile(param_file_new):
                ho_params = pkl.load(open(param_file_new, 'rb'))
                gender = ho_params['gender']
                print("Loading new params from", param_file_new, f'this is the synthesis for shape index {ho_params["index"]}, '
                      f'scan: {smpld_file}, synzer index={self.index}')
        print("Loading params from", self.param_files[self.index])

        pose = np.expand_dims(ho_params['pose'][:72], 0)  # not using hand pose really
        betas = np.expand_dims(ho_params['betas'], 0)
        centers = np.expand_dims(ho_params['trans'], 0)  # body pose, shape from optimized HO params
        offsets = np.expand_dims(smpld_params['v_personal'], 0)  # clothing deformation from randomly sampled one

        random_trans = np.random.uniform(size=3) * 0.7 - 0.35
        random_trans[1] = random_trans[1] * 0.2  # very small y-offset
        global_rot = self.random_rot('y', 360)  # global random rotation

        if "InterCap" in ho_params['sample_frame']:
            # for InterCap, because the camera ray is not parallel to the ground,
            # rotating around y will lead to tilted person, hence use a smaller rotation
            if np.random.uniform() > 0.5:
                global_rot = np.eye(3)
            else:
                # reduce rotation
                global_rot = self.random_rot('y', 180) # smaller rotation angle for intercap

        # apply global rigid transform to smpl
        smpl = SMPLHGenerator.gen_smplh(gender, centers, 1, betas, np.expand_dims(pose[:72], 0))[0]
        smpl_center = self.landmark.get_smpl_center(smpl)
        centers[0] -= smpl_center
        pose, trans = self.global_rot_smpl(pose[0], global_rot, centers[0])  # apply global rotation
        pose, centers = pose[None], trans[None] + smpl_center + random_trans

        # get SMPLD and SMPL mesh
        smpld = SMPLHGenerator.gen_smplh(gender, centers, 1, betas, pose, offsets=offsets)[0]
        smpl = SMPLHGenerator.gen_smplh(gender, centers, 1, betas, pose)[0]

        # get object template based on index
        obj_file, obj_mesh = self.load_obj_mesh(ho_params)

        # get object transformation parameter
        mat = np.eye(4)
        # rigid transform that is optimized by contacts
        mat[:3, :3] = ho_params['obj_rot']
        mat[:3, 3] = ho_params['obj_trans']
        can2frame = ho_params['can2frame']

        mat_comb = mat @ can2frame
        # apply gloabl rigid to obj, first center object, then rotate with same SMPL rigid transform, then translate back
        mat_comb = self.append_transform(mat_comb, -smpl_center)
        mat_comb = self.append_transform(mat_comb, global_rot)
        mat_comb = self.append_transform(mat_comb, random_trans + smpl_center)  # transform back

        ret = {
            'smpl': smpl,
            'smpld': smpld,
            'obj_mesh': obj_mesh,
            'obj_mat': mat_comb,
            "shape_file": obj_file,
            "gender": gender,

            'pose': pose[0],
            'trans': centers,  # TODO: check if there is a bug here!
            'offsets': offsets[0],
            'betas': betas[0],

            "index": ho_params['index'],
            "synset_id": ho_params['synset_id'],
            "ins_name": ho_params['ins_name'],
            "param_file": self.param_files[self.index],
        }

        self.index += 1  # point to next

        return ret

    def load_obj_mesh(self, ho_params):
        """
        load  mesh of the new object shape, without doing any transformation
        this is the same as in synz_base.BaseSynthesizer.load_newshape()
        these meshes are self processed mesh files, they are packed in a tar file
        how these meshes are obtained?
            -shapenet: first waterproof the mesh, then simplify them
            -objaverse and abo: export the mesh from glb file, waterproof and then simplify
        :param ho_params: parameters from synthesized H+O
        :return:
        """
        if self.source == 'shapenet':
            obj_file = osp.join(self.newshape_root, ho_params['synset_id'], ho_params['ins_name'], 'models/model_normalized.obj')
        elif self.source == 'abo':
            obj_file = osp.join(self.newshape_root, 'abo-watertight', ho_params['ins_name'] + "_fused.obj")
        else:
            # this is for /BS/databases24/objaverse/classified-plys
            # obj_file = osp.join(self.newshape_root, ho_params['synset_id'], ho_params['ins_name'] + ".ply")
            obj_file = osp.join(self.newshape_root, ho_params['synset_id'], f'{ho_params["ins_name"]}_fused.obj')
        if self.verbose:
            print("Loading new object shape from", obj_file)
        obj_mesh = Mesh(filename=obj_file)
        obj_mesh = self.simplify_object(obj_mesh, target_face=2500, verbose=self.verbose)
        return obj_file, obj_mesh

    @staticmethod
    def simplify_object(obj: Mesh, target_face=2500, verbose=False) -> Mesh:
        from open3d.cpu.pybind.geometry import TriangleMesh
        from open3d.cpu.pybind.utility import Vector3dVector, Vector3iVector

        verts = Vector3dVector(obj.v)
        faces = Vector3iVector(obj.f)
        if len(faces) < target_face:
            return obj
        mesh_o3d = TriangleMesh(vertices=verts, triangles=faces)
        mesh_simp = mesh_o3d.simplify_quadric_decimation(target_face, 0.5)
        mesh = Mesh(v=np.array(mesh_simp.vertices), f=np.array(mesh_simp.triangles))
        if verbose:
            print(f"Original: {len(verts)} vertices, {len(faces)} faces, simplified: {len(mesh.v)} vertices, {len(mesh.f)} faces.")
        return mesh

    def glob_param_files(self, params_folder):
        param_files = sorted(glob(osp.join(params_folder, '*_params.pkl'))) # return a list of param files
        return param_files

    def copy_mesh(self, mesh):
        m = deepcopy(mesh)
        return m

    def __len__(self):
        return len(self.param_files)

    def get_shape_index(self):
        "get the instance index for current shape"
        ho_params = pkl.load(open(self.param_files[self.index], 'rb'))
        return ho_params['index']  # this is a random value if it is the category with few shapes

    @staticmethod
    def random_rot(axis='y', rmax=360):
        "random rotation around given axis"
        angle = np.random.random() * rmax - rmax / 2
        rot = SynzResultLoader.get_rot_mat(angle, axis)
        return rot

    @staticmethod
    def global_rot_smpl(pose, rot, trans):
        """
        apply global rotation to the smpl pose, after this the parameters should generate a new mesh
        which is equivalent to directly apply this rotation to the original mesh
        :param pose: (72, )
        :param rot: (3, 3) rotation matrix to be applied globally to the full mesh
        :param trans: (3,) SMPL translation parameter
        :return: (72,) new pose under global rotation
        """
        mtx = Rotation.from_rotvec(pose[:3]).as_matrix()
        # the matrix is applied to the vertices, we want another rotation after that, hence it is left multiply
        mtx_new = np.matmul(rot, mtx)
        pose[:3] = Rotation.from_matrix(mtx_new).as_rotvec()
        trans_new = np.matmul(rot, trans)  # translation parameter should also be updated!
        return pose, trans_new

    @staticmethod
    def get_random_rot():
        "generate random rotation matrix"
        rot = Rotation.random()
        return rot.as_matrix()

    @staticmethod
    def get_rot_mat(angle, axis):
        "convert rotation angle around given axis to matrix"
        randian = np.deg2rad(angle)
        rot = np.eye(3)
        if axis == 'y':
            rot[0, 0] = rot[2, 2] = np.cos(randian)
            rot[0, 2] = np.sin(randian)
            rot[2, 0] = -np.sin(randian)
        elif axis == 'x':
            rot[1, 1] = rot[2, 2] = np.cos(randian)
            rot[1, 2] = -np.sin(randian)
            rot[2, 1] = np.sin(randian)
        elif axis == 'z':
            rot[1, 1] = rot[0, 0] = np.cos(randian)
            rot[0, 1] = -np.sin(randian)
            rot[1, 0] = np.sin(randian)
        else:
            raise NotImplementedError
        return rot

    @staticmethod
    def random_global_offset(mean_depth=2.3, xr=0.8, yr=0.15, zr=0.8):
        """
        random global translation for human+object
        x, z direction: extend by 0.5, y direction: 0.3
        :return:
        """
        x = np.random.random() * xr * 2 - xr
        y = np.random.random() * yr * 2 - yr
        z = np.random.random() * zr * 2 - zr + mean_depth
        xyz = np.array([x, y, z])
        return xyz

    def append_transform(self, orig_transform, new_trans: np.ndarray):
        t = np.eye(4)
        if new_trans.ndim == 1:
            t[:3, 3] = new_trans  # translation only
        elif new_trans.ndim == 2:
            t[:3, :3] = new_trans  # rotation
        else:
            raise NotImplementedError
        return np.matmul(t, orig_transform)

    @staticmethod
    def load_smpld_params(file):
        params = pkl.load(open(file, 'rb'), encoding='latin')
        return params

    @staticmethod
    def center_mesh(mesh):
        center = np.mean(mesh.v, 0)
        mesh.v -= center
        return mesh

    def reset_index(self):
        self.index = 0

    def point_next(self):
        self.index += 1

    def set_index(self, index):
        self.index = index