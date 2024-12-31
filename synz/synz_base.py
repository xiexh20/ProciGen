"""
base interaction synthesizer
"""
import sys, os

sys.path.append(os.getcwd())
import numpy as np
from lib_mesh.mesh import Mesh
import igl
import os.path as osp
import trimesh, json
from pytorch3d.renderer import look_at_view_transform

import torch
import pickle as pkl

from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss

from lib_smpl import SMPL_Layer
from synz.interaction_sampler import InteractionSampler
from synz.geometry import GeometryUtils
from synz import viz_utils
import paths


class BaseSynthesizer:
    def __init__(self, seqs_pattern,
                    behave_params_root,
                    smplh_root,
                    obj_temp_path='meshes/chairwood_f50000_clean_cent.obj',
                    corr_mesh_file="meshes/chairblack_f2500_corr.ply",
                    newshape_corr_root="/BS/xxie-2/static00/shapenet/chair-ae8k/",
                    newshape_root="/home/xxie/data/ShapenetV2/",
                    debug=False):
        """

        :param seqs_pattern:
        :type seqs_pattern:
        :param obj_temp_path:
        :type obj_temp_path:
        :param meshes_dir:
        :type meshes_dir:
        :param debug:
        :type debug:
        """
        seqs = InteractionSampler.get_seqs_path(behave_params_root, seqs_pattern)
        self.sampler = InteractionSampler(behave_params_root, seqs, smplh_root, verbose=debug)
        self.smplh_male = SMPL_Layer(model_root=smplh_root,
                                     gender='male', hands=True)
        self.smplh_female = SMPL_Layer(model_root=smplh_root,
                                       gender='female', hands=True)

        # Load object mesh template for the pose sampled from InteractionSampler
        mesh = trimesh.load_mesh(obj_temp_path, process=False)
        self.obj_temp = Mesh(np.array(mesh.vertices), np.array(mesh.faces))
        self.obj_temp.v = self.obj_temp.v - np.mean(self.obj_temp.v, axis=0)
        # o3d_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.obj_temp.v),
        #                                      o3d.utility.Vector3iVector(self.obj_temp.f))
        # self.objv_normals = np.array(o3d_mesh.compute_vertex_normals())  # object vertex normals
        self.corr_v = np.array(trimesh.load_mesh(corr_mesh_file, process=False).vertices)

        # path for new shapes from shapenet/abo/objaverse
        self.newshape_root = newshape_root  # watertight meshes
        self.newshape_corr_root = newshape_corr_root  # path to all corr files, i.e. output from AE
        self.tar_obj_verts = 8000  # desired number of object vertices for optimization
        # map from object class name to corresponding synset name in the corr_root folders
        self.object2synset = {'chair': '03001627',
                              'display': '03211117',
                              'table': '04379243',
                              "trashbin": "02747177",
                              "toolbox": "02773838",
                              "monitor": "03211117",
                              "keyboard": "03085013",
                              "stool": "stool",  # objaverse data
                              "backpack": "backpack",
                              "boxlarge": "box",
                              "boxlong": "box",
                              "boxtiny": "box",
                              "boxsmall": "box",
                              "suitcase": "suitcase",
                              "basketball": "ball",
                              "yogaball": "ball",
                              "box": "box",
                              "ball": "ball",
                              "bottle": ["02876657", "02946921"],  # bottle and can
                              "cup": ["03797390", "02880940"],  # mug and bowl
                              "skateboard": "04225987",
                              "plasticcontainer": "02801938",  # selected from basket category

                              # ABO  objects
                              "abo-chair": 'abo-chair',
                              'abo-table': 'abo-table',  # map to the keys stored in abo-all.json
                              "obja-chair": "obja-chair"

                              }
        self.scan_betas = pkl.load(open(f'{paths.PROCIGEN_ASSET_ROOT}/mgn-scan-betas.pkl', 'rb'))['betas']

        # config for intersection loss
        sigma = 0.5  # The height of the cone used to calculate the distance field loss.
        point2plane = True
        max_collisions = 8  # Increase if get CUDA error about illegal memory.
        hard_meshes = ('box' in obj_temp_path or 'backpack' in obj_temp_path or '08' in obj_temp_path or '06' in obj_temp_path
                    or 'trashbin' in obj_temp_path or 'monitor' in obj_temp_path)
        max_collisions = 20 if hard_meshes else 8
        self.search_tree = BVH(max_collisions=max_collisions)
        linear_max = 0.3 if hard_meshes else 10  # to stabalize optimization
        print("Linear max value:", linear_max, 'max collision:', max_collisions)
        self.pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma,
                                                                         point2plane=point2plane,
                                                                         vectorized=True,
                                                                         linear_max=linear_max
                                                                         )
        self.device = 'cuda:0'

        # debug configuration
        self.render_resolution = (256, 256)
        self.debug = debug
        self.mv = None # do not use psbody.meshviewer

    def get_shape_pool(self, args):
        """
        get the list of shape, synset that can be used to sample from
        :param args:
        :type args:
        :return: list of shapes, corresponding synset id, and synset id of these shapes
        :rtype:
        """
        synset_id = self.object2synset[args.newshape_category]
        new_shape_ids = json.load(open(f'{paths.PROCIGEN_ASSET_ROOT}/new-shape-ids.json'))
        source = args.source
        if args.source == 'shapenet':
            if isinstance(synset_id, list):
                # one object maps to multiple synset/categories
                shapes, synsets = [], []
                for syn in synset_id:
                    shapes_i = sorted(new_shape_ids[source][syn])
                    shapes.extend(shapes_i)
                    synsets.extend([syn] * len(shapes_i))
            else:
                shapes = new_shape_ids[source][synset_id]
                synsets = [synset_id] * len(shapes)
        else:
            shapes = new_shape_ids[source][synset_id]
            synsets = [synset_id] * len(shapes)
        return shapes, synset_id, synsets


    def is_done(self, indices, outfolder, prefix, suffix=''):
        for ind in indices:
            outfile = self.get_outfile(ind, outfolder, prefix)
            if not osp.isfile(outfile):
                return False
        return True

    def get_outfile(self, ind, outfolder, prefix, suffix=''):
        return osp.join(outfolder, f'{prefix}{ind:05d}_params{suffix}.pkl')

    def compute_collision_loss(self, triangles):
        los, N = 0., 5
        for i in range(N): # iterate to reduce randomness
            with torch.no_grad():
                # to find out the collision index
                collision_idxs = self.search_tree(triangles)

            dist = self.pen_distance(triangles, collision_idxs)
            pen_loss = torch.mean(dist)
            los += pen_loss
        return N, los

    def load_newshape(self, args, shape_ind, shapes, synset, num_faces=20000):
        """
        load and simplify new shape meshes
        these meshes are processed mesh files by my self, they are packed in a tar file
        they usually have 15k to 20k faces, adequate enough for optimization
        how these meshes are obtained?
            -shapenet: first waterproof the mesh, then simplify them
            -objaverse and abo: export the mesh from glb file, waterproof and then simplify
        :return:
        :rtype:
        """
        if args.source == 'shapenet':
            file_orig = osp.join(self.newshape_root, synset, shapes[shape_ind], "models/model_normalized_fused.obj")
        elif args.source == 'abo':
            file_orig = osp.join(self.newshape_root, 'abo-watertight', shapes[shape_ind] + "_fused.obj")
        else:
            file_orig = osp.join(self.newshape_root, synset, f'{shapes[shape_ind]}_fused.obj')
        if self.debug:
            print("Loading mesh from", file_orig, shape_ind)
        try:
            # use igl to load
            mm = igl.read_obj(file_orig)
            shape_orig = Mesh(mm[0], mm[3])  # this fix the problem!
        except Exception as e:
            print(e)
            return None
            # continue
        # this can give us always the same number of faces
        shape_orig = GeometryUtils.simplify_object(shape_orig,
                                                   target_face=num_faces)  # simplify mesh to smaller number of vertices
        return shape_orig

    def load_corr_points(self, ins, synset):
        "load correspondence points of new object shape"
        npz_file = osp.join(self.newshape_corr_root, synset, f'{synset}-{ins}.npz')  # this is the same
        shape_corr = np.load(npz_file)['samples']
        return shape_corr

    def compute_newshape_transform(self, sample, shape_corr):
        """
        compute transformation for new object shape from canonical to interaction pose
        :param sample:
        :param shape_corr:
        :return: 4x4 non-rigid transform matrix
        """
        mat, _, _ = trimesh.registration.procrustes(shape_corr, self.corr_v, scale=True)
        pose_obj = np.eye(4)  # rigid object pose
        pose_obj[:3, :3] = sample['obj_R']  # transform from caonincal behave object to interaction pose
        pose_obj[:3, 3] = sample['obj_t']
        mat_comb = np.matmul(pose_obj, mat)
        return mat_comb

    def save_output(self, indices, mat_comb, outfolder, ret_dict, samples, prefix='', suffix=''):
        """

        :param ind:
        :param mat_comb: list of matrices
        :param outfolder:
        :param ret_dict:
        :param sample: list of samples
        :param prefix:
        :param suffix:
        :return:
        """
        for i in range(len(indices)):
            ret_dict_i = {
                'pose': ret_dict['pose'][i],
                'betas': ret_dict['betas'][i],
                'trans': ret_dict['trans'][i],
                'obj_rot': ret_dict['obj_rot'][i],
                'obj_trans':ret_dict['obj_trans'][i],
                # 'synset_id':ret_dict['synset_id'],
                'synset_id': ret_dict['synset_id'][i], # synset id is also a list of ids
                'ins_name': ret_dict['ins_name'][i],
                # "scale":
            }
            self.save_output_single(indices[i], mat_comb[i], outfolder, ret_dict_i,
                                samples[i], prefix, suffix)

    def save_output_single(self, ind, mat_comb, outfolder, ret_dict, sample, prefix='', suffix=''):
        """
        save output parameters
        :param ind: index of the object mesh
        :param mat_comb:
        :param outfolder:
        :param ret_dict:
        :param sample: human gender of the sampled behave frame
        :return:
        """
        pkl.dump({
            "can2frame": mat_comb,  # canonical pose to current frame, for the new object shape
            "pose": ret_dict['pose'],
            "betas": ret_dict['betas'],
            "trans": ret_dict['trans'],
            "gender": sample['gender'],
            "sample_frame":sample['path'],
            "beta_index":sample['beta_index'],

            "pose_old":sample['pose'],
            "betas_old": sample['betas'],
            "trans_old": sample['trans'],

            "obj_rot": ret_dict['obj_rot'], # newly optimized rotation and translation
            "obj_trans": ret_dict['obj_trans'],
            "index": ind,
            "synset_id":ret_dict['synset_id'],
            'ins_name':ret_dict['ins_name'],

            "scale": 1.0 if 'scale' not in ret_dict else ret_dict['scale'],
            "shift": np.zeros((3,)) if 'shift' not in ret_dict else ret_dict['shift']

            # for debug
            # 'cont_dir_orig':ret_dict['cont_dir_orig'],
            # "cont_face_obj":ret_dict["cont_face_obj"],
            # "cont_face_smpl":ret_dict["cont_face_smpl"],
            # 'cont_ind_smpl': ret_dict['cont_ind_smpl'],
            # 'cont_ind_obj': ret_dict['cont_ind_obj']
        }, open(self.get_outfile(ind, outfolder, prefix, suffix), 'wb'))

    def save_original_meshes(self, mask, obj_verts, outfolder, smpl_verts, smpl_verts_inds, prefix=''):
        """

        :param mask: contact mask, 1-in contact
        :param obj_verts: sample points, not vertices anymore
        :param outfolder:
        :param smpl_verts:
        :param smpl_verts_inds:
        :param prefix:
        :return:
        """
        # cmap_cont = self.visu(obj_verts[mask])
        vc_smpl = np.array([[23 / 255., 190 / 255., 207 / 255.]]).repeat(len(smpl_verts), 0)
        vc_obj = np.array([[0.65098039, 0.74117647, 0.85882353]]).repeat(len(obj_verts), 0)
        if np.sum(mask) <= 1:
            # no contact: no color anymore
            cmap_cont = 1.0
            # do not update other vc
        else:
            cmap_cont = self.visu(obj_verts[mask])
            vc_smpl[smpl_verts_inds[:, 0]] = cmap_cont
            vc_obj[mask] = cmap_cont

        # vc_smpl = np.array([[0.65098039, 0.74117647, 0.85882353]]).repeat(len(smpl_verts), 0)
        if self.debug:
            trimesh.Trimesh(smpl_verts, self.smplh_male.faces, vertex_colors=vc_smpl, process=False).export(osp.join(outfolder, f'{prefix}smpl_orig.ply'))
            # Mesh(smpl_verts, self.smplh_male.faces, vc=vc_smpl).write_ply(osp.join(outfolder, f'{prefix}smpl_orig.ply'))
        if self.debug:
            trimesh.PointCloud(obj_verts, colors=vc_obj).export(osp.join(outfolder, f'{prefix}obj_orig.ply'))
            # Mesh(obj_verts, [], vc=vc_obj).write_ply(osp.join(outfolder, f'{prefix}obj_orig.ply'))
            print('Original mesh saved to', outfolder)
        return cmap_cont, vc_obj, vc_smpl

    def render_original_meshes(self, obj_verts, smpl_verts, vc_obj, vc_smpl, obj_faces=None):
        cent = np.mean(np.concatenate([obj_verts, smpl_verts], 0), 0)[None]
        cam_poses = self.get_camera_poses(cent)
        rends = self.render_views(cam_poses,
                                  np.concatenate([obj_faces, self.smplh_male.faces + len(obj_verts)], 0),
                                  np.concatenate([obj_verts, smpl_verts], 0), np.concatenate([vc_obj, vc_smpl], 0),
                                  resolution=self.render_resolution,
                                  )
        return cam_poses, rends

    def get_camera_poses(self, cent):
        return [look_at_view_transform(1.5, ele, azim, at=cent, up=((0., -1., 0),)) for ele, azim in
                zip([-10, -10, -10], [0, 90, 180])]

    def render_results(self, cmap_cont, obj_faces, ret_dict, smpl_verts_inds, vc, cam_poses):
        obj_verts_opt = ret_dict['obj_verts']
        smpl_verts_opt = ret_dict['smpl_verts']
        # render results, for easier result checking

        verts, faces = np.concatenate([obj_verts_opt, smpl_verts_opt], 0), np.concatenate(
            [obj_faces, self.smplh_male.faces + len(obj_verts_opt)], 0)
        vc_smpl = np.array([[23 / 255., 190 / 255., 207 / 255.]]).repeat(len(smpl_verts_opt), 0)
        if len(smpl_verts_inds[:, 0]) > 1:
            # only update if contact
            vc_smpl[smpl_verts_inds[:, 0]] = cmap_cont
        verts_colors = np.concatenate([vc, vc_smpl], 0)
        rends = self.render_views(cam_poses, faces, verts, verts_colors, resolution=self.render_resolution)
        return rends, vc_smpl

    def render_views(self, cam_poses, faces, verts, verts_colors, resolution=(512, 512)):
        rends = []
        for cam in cam_poses:
            R, T = cam
            rend = viz_utils.render_1view(verts, faces, R, T, vc=verts_colors,
                                             light_center=[[0.0, -3.0, 0]],
                                             resolution=resolution)  # change location
            rends.append(rend)
        return rends

    def get_obj_vcmap(self, obj_verts, simp_inds, cmap_cont):
        """
        compute object verts color map
        :param obj_verts:
        :param simp_inds: index to object vertices that are in contact
        :return:
        """
        # cmap_cont = self.visu(obj_verts[simp_inds[:, 0]])
        vc = np.array([[0.65098039, 0.74117647, 0.85882353]]).repeat(len(obj_verts), 0)
        vc[simp_inds[:, 0].tolist()] = cmap_cont  # color contact points differently
        return vc

    @staticmethod
    def visu(vertices):
        """
        compute a color map for all vertices
        Parameters
        ----------
        vertices

        Returns
        -------

        """
        min_coord, max_coord = np.min(vertices, axis=0, keepdims=True), np.max(vertices, axis=0, keepdims=True)
        cmap = (vertices - min_coord) / (max_coord - min_coord)
        return cmap

    @staticmethod
    def sum_dict(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            assert torch.all(~torch.isnan(loss_dict[k])), f'{k} is nan!'
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def copy_smpl_params(self, split_smpl, smpl):
        smpl.pose.data[:, :3] = split_smpl.global_pose.data
        smpl.pose.data[:, 3:66] = split_smpl.body_pose.data
        smpl.pose.data[:, 66:] = split_smpl.hand_pose.data
        smpl.betas.data[:, :2] = split_smpl.top_betas.data

        smpl.trans.data = split_smpl.trans.data

        return smpl