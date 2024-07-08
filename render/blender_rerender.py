"""
re-render ProciGen dataset using blender
key steps:
1. load SMPL-D parameters and transform to interaction space
2. load object shapes and transform to interaction space
3. add textured meshes to blender and render

Author: Xianghui, July 08, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import glob
import json
import sys, os
import time
from trimesh.visual import TextureVisuals

import torch
import trimesh


sys.path.append(os.getcwd())
import bpy
import cv2
from os.path import join, isfile, basename
import numpy as np
import os.path as osp
from tqdm import tqdm
import pickle as pkl

import render.paths as paths
from render.blender_base import BaseRenderer, bpy_version
from render.utils import get_shape_datasetname
from lib_smpl import get_smpl
from render.mesh import MyMesh


class BlenderRerenderer(BaseRenderer):
    def render_seq(self, args):
        "re-render a sequence "
        frames = sorted(glob.glob(args.seq_folder + '/*/'))
        end = len(frames) if args.end is None else args.end 
        frames = frames[args.start:end]
        seq_name, seq_folder = osp.basename(args.seq_folder), args.seq_folder
        dname = get_shape_datasetname(seq_name)
        obj_name = seq_name.split('_')[2]
        seq_out = osp.join(args.out_dir, seq_name+'_rerender')
        os.makedirs(seq_out, exist_ok=True)

        smpld_male, smpld_female = get_smpl('male', True), get_smpl('female', True)
        for frame in tqdm(frames):
            params_obj = pkl.load(open(f'{frame}/{obj_name}/fit01/{obj_name}_fit.pkl', 'rb'))
            params_hum = pkl.load(open(f'{frame}/person/fit01/person_fit.pkl', 'rb'))

            # Prepare object loading
            obj_mat = np.eye(4) # transform matrix for object
            obj_mat[:3, :3] = params_obj['rot']
            obj_mat[:3, 3] = params_obj['trans']
            # load object into blender
            synset_id, ins = params_obj['synset_id'], params_obj['ins_name']
            blender_name_hum, blender_name_obj = 'procigen-hum', 'procigen-obj'
            if dname == 'shapenet':
                # simply apply the parameter
                mesh_file = osp.join(paths.SHAPENET_ROOT, synset_id, ins, 'models/model_normalized.obj')
                bpy.ops.import_scene.obj(filepath=mesh_file, axis_forward='Y', axis_up='Z')
                obj_object = bpy.context.selected_objects[-1]
                obj_object.name = blender_name_obj  # specify the name for the imported one
                bpy.ops.object.shade_smooth()
                obj = bpy.data.objects[blender_name_obj]  # reselect
                obj.select_set(True)
                self.mesh_names.append(blender_name_obj)  # this allows deletion
                self.apply_transform(obj, obj_mat) # now apply transformation to the object
            elif dname == 'objaverse':
                # use objaverse lib to get path
                import objaverse
                uids = [ins]
                # Get local glb file path, if non-existent, will download using uid.
                # Warning: objaverse downloads are saved to home directory by default, if that is not desired,
                # change the BASE_PATH in objaverse.__init__.py file.
                objs = objaverse.load_objects(uids, download_processes=1)
                obj_values = list(objs.values())
                assert len(obj_values) == 1, f'invalid object paths found: {obj_values}'
                glb_path = obj_values[0]
                # Note: the saved object transformation is always w.r.t normalized object meshes
                # Option 1: do normalization inside blender after loading
                self.add_glb_object(glb_path, obj_mat, normalize=True)

                # Option 2: export a normalized glb file, and reload to blender, should also work.
                # export a normalized glb file
                # glb_path_norm = str(glb_path).replace('/glbs/', '/glbs-normalized/')
                # if not osp.isfile(glb_path_norm):
                #     cmd = f'blender -b -P render/blender_export.py -- --object_path {glb_path} --normalize --output_format glb'
                #     os.system(cmd)
                # self.add_glb_object(glb_path_norm, obj_mat, normalize=False)
            else:
                # Note: for abo dataset, the saved object transformation is w.r.t original glb file, w/o further processing.
                glb_path = f'{paths.ABO_ROOT}/{ins}.glb'
                self.add_glb_object(glb_path, obj_mat, normalize=False)

            # Prepare human (SMPLD) vertices
            smpld_model = smpld_female if params_hum['gender'] == 'female' else smpld_male
            smpld_vertices = smpld_model(torch.from_numpy(self.smpl2smplh(params_hum['pose'])[None]).float(),
                                         torch.from_numpy(params_hum['betas'][None]).float(),
                                         torch.from_numpy(params_hum['trans'][None]).float(),
                                         torch.from_numpy(params_hum['offsets'][None]).float())[0]

            scan_name = osp.basename(frame[:-1]).split('-')[1]
            scan_folder = osp.join(paths.MGN_ROOT, scan_name)
            scan_path = paths.ScanPath(scan_folder)
            # load texture uv maps, replacing this with trimesh will result in incorrect uv map
            scan_smpld = MyMesh()
            scan_smpld.load_from_obj(scan_path.smpld_reg_obj())
            scan_smpld.v = smpld_vertices[0].numpy()

            # add human to blender by saving and reloading directly from blender
            frame_folder = osp.join(seq_out, osp.basename(frame[:-1]))
            os.makedirs(frame_folder, exist_ok=True)
            scan_smpld.write_obj(osp.join(frame_folder, f'k1.human.obj'))
            self.add_textured_mesh_from_file(osp.join(frame_folder, f'k1.human.obj'), scan_path.smpld_texture(), blender_name_hum)

            # now render
            bpy_hum = bpy.data.objects[blender_name_hum]
            for k in range(self.camera_count):
                # set output path
                self.output_rgb.file_slots[0].path = join(frame_folder, f'k{k}.color.')
                self.output_depth.file_slots[0].path = join(frame_folder, f'k{k}.depth.')

                # transform human and objects to local camera
                transform = self.cam_transform.world2local_4x4mat(k)
                self.transforom_hum_obj_local(transform)  # both human and object

                # set human to invisible, render object full mask
                bpy_hum.hide_render = True
                bpy.ops.render.render(write_still=True)
                # bpy.ops.wm.save_as_mainfile(filepath=join(frame_folder, f'k{k}.debug.blend'))
                depth_obj = cv2.imread(join(frame_folder, f'k{k}.depth.0001.exr'), cv2.IMREAD_UNCHANGED)[:, :, 0]

                # make human visible again, render human + object
                bpy_hum.hide_render = False
                bpy.ops.render.render(write_still=True)
                depth_full = cv2.imread(join(frame_folder, f'k{k}.depth.0001.exr'), cv2.IMREAD_UNCHANGED)[:, :, 0]

                self.format_outfiles(depth_full, depth_obj, frame_folder, k)
                os.system(f"rm {join(frame_folder, f'k{k}.depth.0001.exr')}")

                # transform back to k1
                self.transforom_hum_obj_local(np.linalg.inv(transform))

            self.reset_scene()
            os.system(f"rm {join(frame_folder, f'k1.human.obj')}")
            self.reinit_light()

    def add_glb_object(self, glb_path, obj_mat, normalize):
        """
        load object from a glb file
        :param glb_path: glb file path
        :param obj_mat: the transformation applied to the object after loading
        :param normalize: normalize the scene or not, for objaverse, it should be normalized
        :return:
        """
        bpy.ops.import_scene.gltf(filepath=glb_path, merge_vertices=True)
        if normalize:
            # this normalization
            self.normalize_scene()  # do not do normalization, since the glb file is already normalized!
        # apply transform to the object
        for obj in self.scene_root_objects():
            if obj.type in ["CAMERA", 'LIGHT']:
                continue  # do not change lighting or camera!
            original_mat = np.array(obj.matrix_world)
            obj.matrix_world = (obj_mat @ original_mat).T

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-s', '--seq_folder', help='path to one ProciGen sequence')
        parser.add_argument('-o', '--out_dir', default='/BS/xxie-6/static00/test')
        parser.add_argument('-fs', '--start', default=0, type=int)
        parser.add_argument('-fe', '--end', default=None, type=int)

        # Render parameters
        parser.add_argument('-resox', type=int, default=2048)
        parser.add_argument('-resoy', type=int, default=1536)

        return parser


if __name__ == '__main__':
    parser = BlenderRerenderer.get_parser()
    args = parser.parse_args()
    icap = 'Date09' in args.seq_folder
    camera_count = 6 if icap else 4
    camera_config = 'assets/icap_cams' if icap else 'assets/behave_cams'

    renderer = BlenderRerenderer(camera_config, camera_count, reso_x=args.resox, reso_y=args.resoy, icap=icap)

    renderer.render_seq(args)


