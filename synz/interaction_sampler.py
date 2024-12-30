"""
sample human object interaction pose from a dataset, e.g. behave
"""

import sys, os
import time

sys.path.append(os.getcwd())
from glob import glob
import numpy as np
from copy import deepcopy
import os.path as osp
import torch
from scipy.spatial.transform import Rotation

from lib_smpl import SMPL_Layer

class InteractionSampler:
    def __init__(self, data_root, seqs, smplh_root, skip_step=6, verbose=False):
        """

        :param seqs: a list of sequences to sample from
        :param data_root: root directory of the parameters
        :param skip_step: skip how many steps for the parameters
        """
        self.seqs = seqs
        self.data_root = data_root

        self.smplh_male = SMPL_Layer(model_root=smplh_root,
                                     gender='male', hands=True)
        self.smplh_female = SMPL_Layer(model_root=smplh_root,
                                       gender='female', hands=True)

        # load parameters
        poses, betas, trans, smpl_vertices, genders = [], [], [], [], []
        obj_R, obj_t = [], []
        paths = []
        for seq in seqs:
            smpl_path = osp.join(data_root, seq, 'smpl_fit_all.npz')
            obj_path = osp.join(data_root, seq, 'object_fit_all.npz')

            packed = np.load(smpl_path, allow_pickle=True)
            # Note: if computing SMPL vertices takes too long, consider pre-compute them and save as files
            layer = self.smplh_male if packed['gender'] == 'male' else self.smplh_female
            verts_gt = layer(torch.from_numpy(packed['poses'][::skip_step]).float(),
                                    torch.from_numpy(packed['betas'][::skip_step]).float(),
                                    torch.from_numpy(packed['trans'][::skip_step]).float())[0].cpu().numpy()
            poses.append(packed['poses'][::skip_step])
            betas.append(packed['betas'][::skip_step])
            trans.append(packed['trans'][::skip_step])
            smpl_vertices.append(verts_gt)
            genders.extend([packed['gender']]*len(verts_gt))

            # object
            packed_obj = np.load(obj_path, allow_pickle=True)
            rots_3x3 = Rotation.from_rotvec(packed_obj['angles'][::skip_step]).as_matrix()
            obj_R.append(rots_3x3)
            obj_t.append(packed_obj['trans'][::skip_step])
            paths.extend([osp.join(seq, str(frame)) for frame in packed['frame_times'][::skip_step]])
        params = {
            "pose": np.concatenate(poses, axis=0),
            'betas': np.concatenate(betas, axis=0),
            'trans': np.concatenate(trans, axis=0),
            'smpl_verts': np.concatenate(smpl_vertices, axis=0),
            'gender': genders,
            'obj_R': np.concatenate(obj_R, axis=0),
            'obj_t': np.concatenate(obj_t, axis=0),
            'path': paths,
        }
        L = len(params['pose'])
        for k, p in params.items():
            assert len(p) == L, f'{k} length is {len(p)} instead of {L}!'
        if verbose:
            print('data loading done, in total {} frames'.format(L))
        self.params = params
        self.verbose = verbose

    def __len__(self):
        return len(self.params['pose'])

    def get_sample(self, shuffle=True):
        """
        randomly sample one data
        :return: 'pose', 'betas', 'trans', 'obj_R', 'obj_t', 'path', 'smpl_verts'
        """
        if shuffle:
            idx = np.random.randint(0, len(self))
        else:
            idx = self.index
            self.index += 1  # sequentially find next
        ret = {}
        for k, v in self.params.items():
            ret[k] = deepcopy(v[idx])
        if self.verbose:
            print('sampled from {}'.format(ret['path']))
        return ret

    @staticmethod
    def get_seqs_path(behave_params_root, pat, train_only=True):
        """
        get abs path to sequences given a path pattern
        :param pat: a pattern to get the sequences from local path
        :param train_only: return only sequences for training
        :return:
        """
        seqs = sorted(glob(osp.join(behave_params_root, pat)))
        if train_only:
            seqs = [x for x in seqs if "Date03" not in x and '_sub09_' not in x and '_sub10_' not in x]
        seqs = [osp.basename(x) for x in seqs]
        return seqs

