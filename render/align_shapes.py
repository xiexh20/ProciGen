"""
simple test function to align object shape in canonical space to interaction space
before usage, please download and modify the dataset paths accordingly
for Objaverse and ABO dataset, blender needs to be installed. The code was tested on blender 2.91.0
"""
import glob
import sys, os

import numpy as np

sys.path.append(os.getcwd())
import trimesh
import os.path as osp
import pickle as pkl
from render.utils import get_shape_datasetname
from render.chamfer_distance import chamfer_distance


SHAPENET_ROOT = '/BS/databases19/ShapeNet/ShapeNetCore.v2' # root path to the shapenet files
OBJAVERSE_ROOT = '/BS/databases24/objaverse' # ROOT path to the objaverse export directory
ABO_ROOT = "/BS/databases23/abo-3dmodels/3dmodels" # root path to all abo glb files
PROCIGEN_ROOT = '/BS/databases24/ProciGen/' # root path to procigen sequences


def test_align(seq_name, num_frames=5):
    ''
    dname = get_shape_datasetname(seq_name)
    obj_name = seq_name.split('_')[2]
    frames = glob.glob(PROCIGEN_ROOT+seq_name+'/*/')[:num_frames]
    for frame in frames:
        param_file = osp.join(frame, obj_name, f'fit01/{obj_name}_fit.pkl')
        params = pkl.load(open(param_file, 'rb'))
        synset_id, ins = params['synset_id'], params['ins_name']
        mesh_inter = trimesh.load(param_file.replace('.pkl', '.ply'), process=False)
        if dname == 'shapenet':
            # simply apply the parameter
            mesh_file = osp.join(SHAPENET_ROOT, synset_id, ins, 'models/model_normalized.obj')
            mesh_can = trimesh.load(mesh_file, process=False, force='mesh')
        elif dname == 'objaverse':
            mesh_file = osp.join(OBJAVERSE_ROOT, f'ply-orig/{ins}.ply')
            if not osp.isfile(mesh_file):
                # load glb file, export as ply file
                import objaverse
                uids = [ins]
                # Get local glb file path, if non-existent, will download using uid.
                objs = objaverse.load_objects(uids, download_processes=1)
                obj_values = list(objs.values())
                assert len(obj_values) == 1, f'invalid object paths found: {obj_values}'
                glb_path = obj_values[0]
                # use blender to export glb file as mesh
                cmd = f'blender -b -P render/blender_export.py -- --object_path {glb_path} --output_path {OBJAVERSE_ROOT} --normalize '
                os.system(cmd)
            # now load mesh
            mesh_can:trimesh.Trimesh = trimesh.load(osp.join(OBJAVERSE_ROOT, f'ply-orig/{ins}.ply'), process=False, force='mesh')

        elif dname == 'abo':
            # get original mesh and transform
            ply_out = osp.join(osp.dirname(ABO_ROOT), 'plys')
            os.makedirs(ply_out, exist_ok=True)
            can_file = osp.join(ply_out, 'ply-orig', f'{ins}.ply')
            if not osp.isfile(can_file):
                glb_path = f'{ABO_ROOT}/{ins}.glb'
                cmd = f'blender -b -P render/blender_export.py -- --object_path {glb_path} --output_path {ply_out}'
                os.system(cmd)
            mesh_can: trimesh.Trimesh = trimesh.load(can_file, process=False, force='mesh')

        # apply transform
        rot, trans = params['rot'], params['trans']
        mesh_can.vertices = np.matmul(mesh_can.vertices, rot.T) + trans

        # sample and compute chamfer
        samples_inter = mesh_inter.sample(100000)
        samples_can = mesh_can.sample(100000)
        cd = chamfer_distance(samples_inter, samples_can)

        # save mesh
        frame_time = osp.basename(frame[:-1])
        print(f"{frame_time} Chamfer distance:", cd)  # this should be very small, up to 1cm.
        mesh_can.export(f'debug/{dname}_{frame_time}_align.ply')
        mesh_inter.export(f'debug/{dname}_{frame_time}_gt.ply')



if __name__ == '__main__':
    test_align('Date04_Subxx_chairwood_synzv2')   # Shapenet seq
    test_align('Date04_Subxx_backpack_batch01', 20)     # Objaverse seq
    test_align('Date04_Subxx_chairwood_abo')      # ABO seq
    print("All done!")