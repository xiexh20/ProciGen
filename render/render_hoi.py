"""
render newly optimized HOI interactions for shapenet objects
"""
import argparse
import glob
import json
import sys, os
import time

sys.path.append(os.getcwd())
import bpy
import cv2
from os.path import join, isfile, basename
import numpy as np
import os.path as osp
from tqdm import tqdm
import pickle as pkl

from lib_mesh.mesh import Mesh
import render.paths as paths
from render.paths import ScanPath
from render.blender_base import BaseRenderer
from render.blender_rerender import BlenderRerenderer
from render.synz_loader import SynzResultLoader


class NewHOIRenderer(BlenderRerenderer):
    def render_folder(self, dname, synz_out, obj_name, start, end, redo, suffix):
        """
        render interaction in a folder, for each frame, randomly select a scan to obtain clothing
        randomly select a scan for synthesize
        :param dname: object shape dataset name
        :param synz_out: output path
        :param scans_root: root path to MGN scans
        :param start:
        :param end:
        :return:
        """
        # use MGN scans
        scan_dataset = paths.MGN_ROOT
        scans = sorted(os.listdir(scan_dataset))

        batch_end = end if end is not None else len(scans)
        seq_out = self.get_seqout_name(obj_name, suffix, synz_out)

        os.makedirs(seq_out, exist_ok=True)

        # iterate over all poses
        self.loader: SynzResultLoader
        self.loader.set_index(start) # point to start
        loop = tqdm(range(start, batch_end))
        loop.set_description(f"{basename(scan_dataset)}:{obj_name}")
        time_start = time.time()
        for idx in loop:
            rid = np.random.randint(len(scans)) # randomly select one scan
            if self.debug:
                rid = idx % len(scans)  # for debug
            scan_folder = join(scan_dataset, scans[rid])
            self.synz_scan(scan_folder, obj_name, seq_out, redo, dname)
        time_end = time.time()
        total_time = time_end - time_start
        print(f"Time to render {len(loop)} frames: {total_time}, average={total_time/len(loop):.4f}")


    def synz_scan(self, scan_folder, obj_name, out_seq, redo, dname):
        """
        synthesize and render N frames using same human scan
        load scan, load synthesize pose, render
        :param scan_folder: a randomly selected human scan
        :param obj_name: the behave object name
        :param out_seq:
        :param redo:
        :param dname: shape dataset name, shapenet/abo/objaverse
        :return:
        """
        scan_path = ScanPath(scan_folder)

        # check if smpld exists
        if not isfile(scan_path.smpld_params()):
            print("no smpld registration for {}".format(scan_folder))
            return

        smpld_reg = self.load_scan_raw(scan_path.smpld_reg_obj())
        if not hasattr(smpld_reg, 'vt'):
            print('no UV map found in {}'.format(scan_path.smpld_reg_obj()))
            return  # check if the mesh contains vertex uv map

        # one scan-per shape
        frame = f'{self.loader.get_shape_index():05d}-{scan_path.name}'
        frame_folder = join(out_seq, frame)
        frames = glob.glob(join(out_seq, f'{self.loader.get_shape_index():05d}-*'))
        if len(frames)> 0:
            if self.is_done(frames[0]) and not redo:
                print(f'{frame_folder} done, skipped')
                self.loader.point_next() # do not forget to point to next!
                return

        if self.is_done(frame_folder) and not redo:
            print(f'{frame_folder} done, skipped')
            self.loader.point_next()
            return

        ret = self.loader.load(scan_path.smpld_params())
        smpl, smpld, obj_mesh = ret['smpl'], ret['smpld'], ret['obj_mesh']
        ret['scan_path'] = scan_path.folder

        smpld_reg.v = smpld.v
        smpld_reg.f = smpld.f

        # save mesh results and smpl, obj parameters
        self.save_synz(ret, frame_folder, obj_mesh, obj_name, smpld_reg)

        # render image and masks
        obj_params = {
            'rot': ret['obj_mat'][:3, :3],
            'trans': ret['obj_mat'][:3, 3],
            'synset_id': ret['synset_id'],
            'ins_name': ret['ins_name']
        }
        self.render_frame(dname, frame_folder, obj_params, scan_path, smpld_reg)

    @staticmethod
    def load_scan_raw(file: str, center=False):
        scan_raw = Mesh()
        if file.endswith('.obj'):
            scan_raw.load_from_obj(file)
        else:
            scan_raw.load_from_ply(file)
        if center:
            c = np.mean(scan_raw.v, 0)
            scan_raw.v -= c

        return scan_raw

    def is_done(self, frame_folder):
        pats = [f'k*.color.{self.ext}', f'k*.person_mask.png', f'k*.obj_rend_mask.png']
        for pat in pats:
            if not self.check_files(join(frame_folder, pat)):
                return False
        return True

    def check_files(self, pat):
        files = glob.glob(pat)
        return len(files) == self.camera_count

    def save_synz(self, ret, frame_folder, obj_simplified, obj_name, smpld_reg=None):
        """

        :param ret: ret dict from synthesizer
        :param frame_folder:
        :param obj_simplified: object template, to be saved after applying transformation here
        :param obj_name: behave object class name
        :param smpld_reg: SMPLD mesh
        :return:
        """
        save_name = 'fit01'
        os.makedirs(frame_folder, exist_ok=True)
        person_folder = join(frame_folder, 'person')
        os.makedirs(person_folder, exist_ok=True)
        # smpld_reg.write_ply(join(person_folder, 'person.ply')) # psuedo point cloud
        os.makedirs(join(person_folder, save_name), exist_ok=True)
        smpl = ret['smpl']
        smpl.write_ply(join(person_folder, save_name, 'person_fit.ply'))
        obj_folder = join(frame_folder, obj_name, save_name)
        os.makedirs(obj_folder, exist_ok=True)
        obj_R, obj_t = ret['obj_mat'][:3, :3], ret['obj_mat'][:3, 3]
        obj_simplified.v = np.matmul(obj_simplified.v, obj_R.T) + obj_t
        obj_simplified.write_ply(join(obj_folder, f'{obj_name}_fit.ply'))

        # save parameters
        outfile = join(person_folder, save_name, 'person_fit.pkl')
        pkl.dump({
            'pose': ret['pose'],
            'betas': ret['betas'],
            'offsets': ret['offsets'],
            'trans': ret['trans'],
            "param_file":ret['param_file'],
            "scan_path":ret['scan_path'],
            "gender":ret['gender'],

            "scale": ret['scale'] if 'scale' in ret else 1.0, # this is for normalized rendering
            "center": ret['center'] if 'center' in ret else None,
            "mean_center": ret['mean_center'] if 'mean_center' in ret else None
        }, open(outfile, 'wb'))

        # object parameters
        outfile = join(obj_folder, f'{obj_name}_fit.pkl')
        pkl.dump({
            "rot": obj_R, 'trans': obj_t,
            "param_file":ret['param_file'],
            "synset_id":ret['synset_id'],
            "ins_name":ret['ins_name'],
            "index":ret['index'],

            # normalization parameters
            "scale": ret['scale'] if 'scale' in ret else 1.0,
            "center": ret['center'] if 'center' in ret else None,
            "mean_center": ret['mean_center'] if 'mean_center' in ret else None
        }, open(outfile, 'wb'))
        print('saved synz parameters to', outfile)

    def get_seqout_name(self, obj_name, suffix, synz_out):
        if 'behave' in self.camera_config:
            seq_out = join(synz_out, f'Date04_Subxx_{obj_name}_{suffix}')
        elif self.icap:
            seq_out = join(synz_out, f'Date09_Subxx_{obj_name}_{suffix}')
        else:
            raise ValueError(f'Please specify a name pattern for newly synthesized sequences')
        return seq_out

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--params_folder', required=True, help='folder to optimized parameters')
        parser.add_argument('-c', '--camera_config', default='assets/behave_cams')
        parser.add_argument('-fs', '--start', default=0, type=int)
        parser.add_argument('-fe', '--end', default=None, type=int)
        parser.add_argument('-src', '--source', choices=['objaverse', 'shapenet', 'abo'], default='shapenet',
                            help='source dataset of the new objects')

        # output config
        parser.add_argument('-o', '--out_dir', default='outputs/render')
        parser.add_argument('-on', '--obj_name', required=True, help='behave object name')
        parser.add_argument('-suffix', default='01', help='suffix for the generated sequence folder')

        parser.add_argument('-redo', default=False, action='store_true')

        parser.add_argument('-resox', type=int, default=2048//2)
        parser.add_argument('-resoy', type=int, default=1536//2)

        # intercap setting or behave
        parser.add_argument('-icap', default=False, action='store_true')

        return parser



def main():
    args = NewHOIRenderer.get_parser().parse_args()
    camera_count = 6 if args.icap else 4
    if args.icap:
        args.resox = 1920//2
        args.resoy = 1080//2

    if args.source == 'shapenet':
        newshape_root = paths.SHAPENET_ROOT
    elif args.source == 'abo':
        newshape_root = paths.ABO_MESHES_ROOT
    else:
        newshape_root = paths.OBJAVERSE_MESHES_ROOT

    loader = SynzResultLoader(args.params_folder, newshape_root, paths.PROCIGEN_ASSET_ROOT)
    renderer = NewHOIRenderer(args.camera_config, camera_count, loader=loader,
                              reso_x=args.resox, reso_y=args.resoy, icap=args.icap)
    renderer.render_folder(args.source, args.out_dir, args.obj_name,
                           args.start, args.end, args.redo, args.suffix)
    print('all done')


if __name__ == '__main__':
    main()


