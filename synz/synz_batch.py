"""
batchfied synthesis of interaction with new shapes
"""

import sys, os
sys.path.append(os.getcwd())

import time
import torch
import os.path as osp
from sklearn.neighbors import KDTree
import numpy as np
import igl, cv2
from tqdm import tqdm
import imageio
from torch.optim import Adam
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes

from lib_smpl.smpl_generator import SMPLHGenerator
from lib_smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from lib_mesh.mesh import Mesh
from synz.synz_base import BaseSynthesizer


class BatchSynthesizer(BaseSynthesizer):
    def synthesize(self, args):
        """
        main entry to synthesis/optimization
        :param args:
        :type args:
        :return:
        :rtype:
        """
        start, end = args.start, args.end
        shapes, synset_id, synsets = self.get_shape_pool(args)

        # these objects have too few shapes, randomly sample them instead of sequentially synthesize
        random_shape = (synset_id in ['box', 'backpack', 'suitcase', 'stool']
                        or args.object == 'yogaball' or args.random_shape or
                        args.object_category in ['box', 'backpack', 'suitcase', 'stool', 'cup', 'ball',
                                                 'toolbox', 'bottle', 'skateboard', 'plasticcontainer', 'keyboard',
                                                 'trashbin', 'monitor'])
        end_count = len(self.sampler) if random_shape else len(shapes)
        end = end_count if end is None else end
        if self.debug:
            print("Using random object shape?", random_shape)

        outfolder = args.outfolder
        os.makedirs(outfolder, exist_ok=True)
        REDO = False

        # for visualization
        video_writer = imageio.get_writer(osp.join(outfolder, f'{osp.basename(outfolder)}_synz_{start}-{end}.mp4'),
                                          format='FFMPEG', mode='I', fps=1)
        batch_size = args.batch_size
        frames = range(start, end, batch_size)
        for bidx in tqdm(frames):
            batch_start = time.time()
            if random_shape:
                shape_indices = np.random.choice(len(shapes), batch_size)
            else:
                shape_indices = np.arange(bidx, min(bidx+batch_size, len(shapes)))

            batch_size = len(shape_indices)  # update batch size to the number of shapes
            if self.is_done(np.arange(bidx, bidx + batch_size), outfolder, '') and not REDO:
                # load rendered images and append to video
                for i in range(bidx, bidx + batch_size):
                    rend_file = osp.join(outfolder, f'{i:05d}_obj.jpg')
                    if osp.isfile(rend_file):
                        img = cv2.imread(rend_file)
                        video_writer.append_data(img[:, :, ::-1])
                continue

            # Step 1: sample interaction
            samples = []
            for i in range(batch_size):
                sample = self.sampler.get_sample()
                # also randomly sample a beta from scan dataset
                beta_index = np.random.randint(0, len(self.scan_betas))
                betas_random = self.scan_betas[beta_index].copy()
                sample['betas'] = betas_random
                sample['beta_index'] = beta_index
                samples.append(sample)

            # compute contacts for each interaction sample
            poses, betas, trans = [], [], []
            newshape_transforms = []  # transformation init for new shapes
            # contacts information
            cont_inds_objs, cont_inds_smpls = [], []
            cont_faces_objs, cont_faces_smpls = [], []
            shape_indices_cont, out_indices = [], []
            bary_coords_objs, bary_coords_smpls = [], []
            bary_verts_ids_objs, bary_verts_ids_smpls = [], []
            verts_objs, faces_objs = [], []  # vertex and faces of new shapes
            cont_dirs = []
            cont_obj_masks = []
            gender = 'male' if np.random.random() > 0.5 else 'female'

            start = time.time()
            for i, (shape_ind, sample) in enumerate(zip(shape_indices, samples)):
                sample['gender'] = gender  # update the gender information to be saved in dict
                ins = shapes[shape_ind]  # shpenet/objaverse uid
                obj_shape_new = self.load_newshape(args, shape_ind, shapes, synsets[shape_ind], num_faces=20000)

                newshape_corr = self.load_corr_points(ins, synsets[shape_ind])
                # compute alignment to canonical mesh
                mat_comb = self.compute_newshape_transform(sample, newshape_corr)
                newshape_corr = np.matmul(newshape_corr, mat_comb[:3, :3].T) + mat_comb[:3, 3]
                obj_shape_new.v = np.matmul(obj_shape_new.v, mat_comb[:3, :3].T) + mat_comb[:3, 3]
                # compute template mesh in interaction pose
                obj_points_behave = np.matmul(self.corr_v, sample['obj_R'].T) + sample['obj_t']

                # transfer contact based on closest points
                smpl_verts = sample['smpl_verts']
                dist, face_inds, points_smpl = igl.signed_distance(obj_points_behave, smpl_verts, self.smplh_male.faces)
                # increase to 0.22 to use only more than 2 points
                mask_obj_pts = np.abs(dist) < 0.022  # these points are in contact, transfer to a new shape
                if np.sum(mask_obj_pts) <= 1:  # only one point will have bug
                    print(f"No contact for {sample['path']}!")
                    # save original pose parameters
                    ret_dict = {
                        "pose": sample['pose'],
                        "betas": sample['betas'],
                        "trans": sample['trans'],
                        "obj_rot": np.eye(3),
                        "obj_trans": np.zeros((3,)),

                        "synset_id": synsets[shape_ind],
                        "ins_name": ins

                    }
                    self.save_output_single(bidx + i, mat_comb, outfolder, ret_dict, sample)

                    if args.two_gender:
                        gender_other = 'male' if sample['gender'] == 'female' else 'female'
                        sample['gender'] = gender_other
                        self.save_output_single(bidx + i, mat_comb, outfolder, ret_dict, sample, suffix='_other_gender')
                    continue

                num_contact = np.sum(mask_obj_pts)
                cont_points_smpl = points_smpl[mask_obj_pts]  # SMPL surface points, reshape to make sure it is 2d array
                cont_points_obj = obj_points_behave[mask_obj_pts]  # object points in contact
                cont_dir_orig = cont_points_obj - cont_points_smpl
                cont_dir_orig = cont_dir_orig / np.linalg.norm(cont_dir_orig, axis=1)[:, None]

                # contact SMPL vertices (not used in computing loss)
                tree = KDTree(smpl_verts)
                _, smpl_verts_inds = tree.query(cont_points_obj)
                newobjv_dense, newobjf_dense = np.array(obj_shape_new.v), np.array(obj_shape_new.f).astype(int)

                # Transfer contacts to new shape points
                cont_points_obj_newshape = newshape_corr[mask_obj_pts]
                tree_obj = KDTree(newobjv_dense)
                _, cont_verts_inds_obj = tree_obj.query(cont_points_obj_newshape)

                # Transfer contacts to new object shape mesh
                _, cont_faces_obj, obj_cont_surface_points = igl.signed_distance(cont_points_obj_newshape,
                                                                                 newobjv_dense,
                                                                                 newobjf_dense.astype(int))
                dist, cont_faces_smpl, smpl_cont_surface_points = igl.signed_distance(cont_points_obj, smpl_verts,
                                                                                      self.smplh_male.faces)
                # find barycentric coordinates for in new object mesh
                bary_verts_ids_obj, bary_coords_obj = obj_shape_new.barycentric_coordinates_for_points(
                    obj_cont_surface_points, cont_faces_obj)
                bary_verts_ids_smpl, bary_coords_smpl = Mesh(smpl_verts,
                                                             self.smplh_male.faces).barycentric_coordinates_for_points(
                    smpl_cont_surface_points,
                    cont_faces_smpl)

                # save for optimization
                poses.append(torch.from_numpy(sample['pose']).to(self.device))
                betas.append(torch.from_numpy(sample['betas']).to(self.device))
                trans.append(torch.from_numpy(sample['trans']).to(self.device))
                cont_inds_objs.append(torch.from_numpy(cont_verts_inds_obj[:, 0]).to(self.device))
                cont_inds_smpls.append(torch.from_numpy(smpl_verts_inds[:, 0]).to(self.device))
                cont_faces_objs.append(torch.from_numpy(newobjf_dense[cont_faces_obj]).long().to(self.device))
                cont_faces_smpls.append(torch.from_numpy(self.smplh_male.faces[cont_faces_smpl]).long().to(self.device))
                bary_coords_objs.append(torch.from_numpy(bary_coords_obj).to(self.device))
                bary_coords_smpls.append(torch.from_numpy(bary_coords_smpl).to(self.device))
                bary_verts_ids_objs.append(torch.from_numpy(bary_verts_ids_obj.astype(int)).to(self.device))
                bary_verts_ids_smpls.append(torch.from_numpy(bary_verts_ids_smpl.astype(int)).to(self.device))
                verts_objs.append(torch.from_numpy(newobjv_dense).float().to(self.device))
                faces_objs.append(torch.from_numpy(newobjf_dense.astype(int)).to(self.device))
                cont_dirs.append(torch.from_numpy(cont_dir_orig).to(self.device))

                shape_indices_cont.append(shape_ind)  # shapes with contact for optimization
                newshape_transforms.append(mat_comb)
                out_indices.append(i + bidx)
                cont_obj_masks.append(mask_obj_pts)

                if self.debug:
                    _, vc_obj, vc_smpl = self.save_original_meshes(mask_obj_pts,
                                                                   obj_points_behave, outfolder,
                                                                   smpl_verts,
                                                                   smpl_verts_inds)

            if len(shape_indices_cont) == 0:
                print("No contact for this batch, skipped")
                continue
            end = time.time()
            if self.debug:
                print(f"Contact computation done after {end - start:.4f} seconds of {len(shape_indices_cont)} samples")
            # data for batchified optimization
            data_dict = {
                "pose": torch.stack(poses, 0),
                'betas': torch.stack(betas, 0),
                "trans": torch.stack(trans, 0),
                'gender': gender,  # randomly selected TODO: check number of mismatches scan and genders

                "obj_verts": verts_objs,
                "obj_faces": faces_objs,
                "obj_meshes": Meshes(verts_objs, faces_objs),
                "smpl_faces": torch.from_numpy(self.smplh_male.faces).long().to(self.device),

                # contacts
                "cont_ind_obj": cont_inds_objs,
                "cont_ind_smpl": cont_inds_smpls,
                'cont_dir_orig': cont_dirs,
                "cont_face_obj": cont_faces_objs,
                'cont_face_smpl': cont_faces_smpls,
                "bary_coords_smpl": bary_coords_smpls,
                "bary_coords_obj": bary_coords_objs,
                "bary_verts_ids_smpl": bary_verts_ids_smpls,
                "bary_verts_ids_obj": bary_verts_ids_objs,

            }
            data_dict = self.process_data_dict(data_dict) # process for batch operation

            ret_dict = self.optimize_smpl_obj(data_dict, args.iterations)
            # try:
            #     ret_dict = self.optimize_smpl_obj(data_dict, args.iterations)
            # except Exception as e:
            #     print("Optimization failed with error:", e)
            #     continue

            ret_dict['synset_id'] = [synsets[sid] for sid in shape_indices_cont]
            ret_dict['ins_name'] = [shapes[x] for x in shape_indices_cont]
            samples_cont = [samples[x - bidx] for x in out_indices]
            self.save_output(out_indices, newshape_transforms, outfolder, ret_dict, samples_cont)

            if args.two_gender: # for each new frame, optimize pose for both genders
                gender_other = 'male' if gender == 'female' else 'female'
                data_dict['gender'] = gender_other
                try:
                    ret_dict = self.optimize_smpl_obj(data_dict, args.iterations)
                except Exception as e:
                    print("Optimization failed with error:", e)
                    continue
                ret_dict['synset_id'] = [synsets[sid] for sid in shape_indices_cont]
                ret_dict['ins_name'] = [shapes[x] for x in shape_indices_cont]
                for sample in samples_cont:
                    sample['gender'] = gender_other
                self.save_output(out_indices, newshape_transforms, outfolder, ret_dict, samples_cont, suffix='_other_gender')


            batch_end = time.time()
            total_time = batch_end - batch_start
            if self.debug:
                print(f"Total time for batch of {batch_size} frames: {total_time:.4f}, average: {total_time / batch_size:.4f}")

            # Visualization, disable this if it is too slow
            for i, sample in enumerate(samples_cont):
                obj_verts_orig = np.matmul(self.obj_temp.v, sample['obj_R'].T) + sample['obj_t']
                smpl_verts = sample['smpl_verts']
                dist_orig, _, _ = igl.signed_distance(obj_verts_orig, smpl_verts, self.smplh_male.faces)
                # np.abs(dist_orig) < 0.02
                mask_cont_orig = np.abs(dist_orig) < 0.02
                if np.sum(mask_cont_orig) == 0:
                    # dummy vis, all contacts
                    mask_cont_orig[:] = True
                cmap_cont_orig = self.visu(obj_verts_orig[mask_cont_orig])
                vc_obj_orig = np.array([[0.65098039, 0.74117647, 0.85882353]]).repeat(len(obj_verts_orig), 0)
                vc_obj_orig[mask_cont_orig] = cmap_cont_orig

                mask = cont_obj_masks[i]
                obj_points_behave = np.matmul(self.corr_v, sample['obj_R'].T) + sample['obj_t']

                smpl_verts_inds = cont_inds_smpls[i].cpu().numpy()[:, None] # (N,) -> (N, 1) for backward compatibility
                cont_verts_inds_obj = cont_inds_objs[i].cpu().numpy()[:, None]
                objv_dense = verts_objs[i].cpu().numpy()
                objf_dense = faces_objs[i].cpu().numpy()

                cmap_cont, vc_obj, vc_smpl = self.save_original_meshes(mask, obj_points_behave, outfolder, smpl_verts,
                                                                       smpl_verts_inds)

                cam_poses, rends_orig = self.render_original_meshes(obj_verts_orig, smpl_verts, vc_obj_orig, vc_smpl,
                                                                    self.obj_temp.f)

                vc_obj = self.get_obj_vcmap(objv_dense, cont_verts_inds_obj, cmap_cont)
                ret_dict_i = {'obj_verts':ret_dict['obj_verts'][i], 'smpl_verts':ret_dict['smpl_verts'][i]}
                rends, vc_smpl = self.render_results(cmap_cont, objf_dense, ret_dict_i, smpl_verts_inds, vc_obj, cam_poses)
                rend_comb = np.concatenate([np.concatenate(rends_orig, 1), np.concatenate(rends, 1)], 0)
                frame = f"{osp.basename(osp.dirname(sample['path']))}-{osp.basename(sample['path'])}-{out_indices[i]}"
                cv2.putText(rend_comb, frame, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                cv2.imwrite(osp.join(outfolder, f'{out_indices[i]:05d}_obj.jpg'), rend_comb[:, :, ::-1])
                video_writer.append_data(rend_comb)
            end = time.time()
            # Time to visualize 128 frames: 13.13868522644043
            if self.debug:
                print(f"Time to visualize {batch_size} frames:", end - batch_end)

    def optimize_smpl_obj(self, data_dict, num_steps=300):
        """
        optimize SMPL and object parameters transferring to new shape
        :param data_dict:
        :type data_dict:
        :param num_steps:
        :type num_steps:
        :return:
        :rtype:
        """
        device = self.device
        # adapt for batchified version
        assert data_dict['pose'].ndim == 2, 'the input pose is not batchfied version!'
        smpl = SMPLHGenerator.get_smplh(data_dict['pose'],
                                        data_dict['betas'],
                                        data_dict['trans'], data_dict['gender'])
        bs = data_dict['pose'].shape[0]
        # split and optimize only body pose
        smpl_split = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl)

        obj_angles = torch.zeros(bs, 3).to(device).requires_grad_(True)
        obj_trans = torch.zeros(bs, 3).to(device).requires_grad_(True)

        smpl_optimizer = Adam([smpl_split.body_pose, smpl_split.trans, smpl_split.top_betas], lr=0.003, weight_decay=0.1)
        obj_optimizer = Adam([obj_angles, obj_trans], lr=0.005,
                             weight_decay=0.1)  # a bit larger learning rate for objects


        with torch.no_grad():
            smpl_verts, _, _, _ = smpl()
            data_dict["pinit"] = smpl.pose.clone()
            data_dict['tinit'] = smpl.trans.clone()
            if self.debug:
                cmap = self.visu(smpl_verts.cpu().numpy()[0, data_dict['cont_ind_smpl'][0].cpu().numpy()])

        # num_steps = 300
        loop = tqdm(range(num_steps), desc='optimizing SMPL+object')
        loss_weights = self.get_loss_weights()
        for it in loop:
            smpl_optimizer.zero_grad()
            obj_optimizer.zero_grad()

            loss_dict, obj_verts_new, smpl_verts = self.compute_loss(data_dict, obj_angles, obj_trans, smpl_split)
            # decay = it/50. this is reasonable, 100 is too small
            decay = it / 50.
            l_str = 'Iter: {}'.format(it)
            for k in loss_dict:
                v = loss_weights[k](loss_dict[k], decay).mean().item()
                # print(f"Iter {it} loss {k} = {v:.4f}")
                l_str += ', {}: {:0.4f}'.format(k, v)
            loop.set_description(l_str)

            loss = self.sum_dict(loss_dict, loss_weights, decay)

            loss.backward()
            obj_optimizer.step()
            smpl_optimizer.step()

            if self.mv is not None:
                ind = 0
                vc_smpl = np.array([[1., 0, 0]]).repeat(smpl_verts.shape[1], 0)
                vc_smpl[data_dict['cont_ind_smpl'][ind].cpu().numpy()] = cmap
                smpl_mesh = Mesh(smpl_verts[ind].detach().cpu().numpy(), [], vc=vc_smpl)
                vc_obj = np.array([[0., 1., 0.]]).repeat(obj_verts_new.shape[1], 0)
                vc_obj[data_dict['cont_ind_obj'][ind].cpu().numpy()] = cmap
                obj_mesh = Mesh(obj_verts_new[0].detach().cpu().numpy(), [], vc=vc_obj)
                self.mv.set_static_meshes([smpl_mesh, obj_mesh])

        # return optimized results
        with torch.no_grad():
            # add a random global translation to the SMPL and objects
            add_global_offset = False if 'add_global_offset' not in data_dict else data_dict['add_global_offset']
            if add_global_offset: # for ProciGen-Video
                random_trans = np.random.uniform(size=3) * 0.8 - 0.4
                random_trans[1] = random_trans[1] * 0.2 # very small y-offset
                print(f"Adding random global offset {random_trans} to human + object")
                random_trans = torch.from_numpy(random_trans[None]).float().repeat(len(obj_angles), 1).to(device)
                smpl_split.trans += random_trans
                obj_trans += random_trans

            smpl_verts, _, _, _ = smpl_split()
            rotations = axis_angle_to_matrix(obj_angles)
            obj_verts_new = [torch.bmm(v[None], rot[None].transpose(1, 2)) + t[None].unsqueeze(1)
                             for v, rot, t in zip(data_dict['obj_meshes'].verts_list(), rotations, obj_trans)]
            obj_verts_new = [v.detach().cpu().numpy()[0] for v in obj_verts_new]  # a list of [N, 3]

        # copy smpl_split parameters
        smpl = self.copy_smpl_params(smpl_split, smpl)

        ret_dict = {
            "smpl_verts": smpl_verts.detach().cpu().numpy(),
            'pose': smpl.pose.detach().cpu().numpy(),
            'betas': smpl.betas.detach().cpu().numpy(),
            'trans': smpl.trans.detach().cpu().numpy(),
            'obj_verts': obj_verts_new,  # for rendering, make sure the vertices have same number as original mesh
            'obj_rot': rotations.detach().cpu().numpy(),
            'obj_trans': obj_trans.detach().cpu().numpy(),
        }
        return ret_dict

    def compute_loss(self, data_dict, obj_angles, obj_trans, smpl):
        "replace for loop with packed indexing"
        smpl_verts, _, _, _ = smpl()
        rotations = axis_angle_to_matrix(obj_angles)
        obj_meshes: Meshes = data_dict['obj_meshes']
        obj_verts_padded = obj_meshes.verts_padded()
        verts_obj_new = torch.bmm(obj_verts_padded, rotations.transpose(1, 2)) + obj_trans.unsqueeze(1)
        # obj_meshes.verts_list()
        # verts_count = obj_meshes.num_verts_per_mesh()
        # obj_verts_new = [v[:x] for v, x in zip(obj_verts_new, verts_count)]

        loss_pinit = torch.mean((smpl.pose - data_dict["pinit"]) ** 2)
        loss_tinit = torch.mean((smpl.trans - data_dict['tinit']) ** 2)

        # collision loss
        loss_colli = self.collision_loss_batch(data_dict, obj_meshes, smpl_verts, verts_obj_new)

        loss_dict = {
            "normal": 0.,
            "cdir": 0,
            "cont": 0.,
            # 'colli': torch.clamp(loss_colli, -10, 100.),
            'colli': loss_colli,
            "tinit": loss_tinit,
            'pinit': loss_pinit,
        }

        # contact surface points based on barycentric coordinates
        self.compute_contact_losses(data_dict, loss_dict, smpl_verts, verts_obj_new)

        # for k, loss in loss_dict.items():
        #     if torch.all(~torch.isnan(loss)):
        #         print()
        return loss_dict, verts_obj_new, smpl_verts

    def compute_contact_losses(self, data_dict, loss_dict, smpl_verts, verts_obj_new):
        "a set of contact based losses"
        verts_smpl_packed = smpl_verts.reshape([-1, 3])  # (N, 3)
        verts_obj_packed = verts_obj_new.reshape([-1, 3])  # (N, 3)

        # use already preprocessed ones:
        bary_verts_ids_smpl = data_dict['bary_verts_ids_smpl_packed']
        bary_verts_ids_obj = data_dict['bary_verts_ids_obj_packed']
        bary_coords_smpl = data_dict['bary_coords_smpl_packed']
        bary_coords_obj = data_dict['bary_coords_obj_packed']
        cont_face_smpl = data_dict['cont_face_smpl_packed']
        cont_face_obj = data_dict['cont_face_obj_packed']
        cont_dir_orig = data_dict['cont_dir_orig_packed']
        per_point_weights = data_dict['per_point_weights']  # (N,)
        cont_smpl = (verts_smpl_packed[bary_verts_ids_smpl] * bary_coords_smpl).sum(1)
        cont_obj = (verts_obj_packed[bary_verts_ids_obj] * bary_coords_obj).sum(1)

        loss_cont = (torch.sum((cont_obj - cont_smpl) ** 2, -1) * per_point_weights).sum()
        loss_dict['cont'] = loss_cont
        # surface normals loss
        d1_smpl = verts_smpl_packed[cont_face_smpl[:, 1]] - verts_smpl_packed[cont_face_smpl[:, 0]]
        d2_smpl = verts_smpl_packed[cont_face_smpl[:, 2]] - verts_smpl_packed[cont_face_smpl[:, 0]]
        normal_smpl = torch.cross(d1_smpl, d2_smpl, dim=1)  # (N, 3)x (N, 3) -> (N, 3)
        normal_len_smpl = torch.linalg.norm(normal_smpl, dim=1)[:, None]
        normal_smpl = normal_smpl / (normal_len_smpl + 1e-6)
        # maskn_smpl = normal_len_smpl > 1e-4  # (N, 1)
        maskn_smpl = ~torch.isnan(normal_smpl)
        d1_obj = verts_obj_packed[cont_face_obj[:, 1]] - verts_obj_packed[cont_face_obj[:, 0]]
        d2_obj = verts_obj_packed[cont_face_obj[:, 2]] - verts_obj_packed[cont_face_obj[:, 0]]
        normal_obj = torch.cross(d1_obj, d2_obj, dim=1)  #
        normal_len = torch.linalg.norm(normal_obj, dim=1)[:, None]
        normal_obj = normal_obj / (normal_len + 1e-6)
        maskn_obj = ~torch.isnan(normal_obj)
        mask_normal = maskn_smpl & maskn_obj
        N = normal_obj.shape[0]
        prod = torch.matmul(normal_smpl.reshape(N, 1, 3), normal_obj.reshape(N, 3, 1)).squeeze(1)
        loss_dict['normal'] = torch.sum((prod + 1.) * per_point_weights[:, None] * mask_normal.float())  # consider only valid normals

        # contact direction loss
        cont_dir = cont_obj - cont_smpl
        cont_dir_len = torch.linalg.norm(cont_dir, dim=1)[:, None]
        cont_dir = cont_dir / cont_dir_len

        prod = torch.matmul(cont_dir.reshape(N, 1, 3), cont_dir_orig).squeeze(1)
        loss_dict['cdir'] = torch.sum((1. - prod) * per_point_weights[:, None] * (cont_dir_len).float())

    def collision_loss_batch(self, data_dict, obj_meshes, smpl_verts, verts_obj_new):
        smpl_verts_count = smpl_verts.shape[1]
        comb_verts = torch.cat([smpl_verts, verts_obj_new], 1)
        obj_faces = obj_meshes.faces_padded()
        comb_faces = [torch.cat([data_dict['smpl_faces'].clone(), of + smpl_verts_count], 0) for of in
                      obj_faces]  # (F, 3)]
        bs, nv = comb_verts.shape[:2]
        # additional offset added to each mesh instanc in the batch
        faces_idx = torch.stack(comb_faces, 0) + \
                    (torch.arange(bs, dtype=torch.long).to(self.device) * nv)[:, None, None]
        triangles = comb_verts.reshape([-1, 3])[faces_idx]  # (B, T, 3, 3)
        N, los = self.compute_collision_loss(triangles)
        loss_colli = los / N
        return loss_colli

    def get_loss_weights(self):
        loss_weight = {
            'cont': lambda cst, it: 20. ** 2 * cst / (1 + it),
            'colli': lambda cst, it: 3. ** 2 * cst / (1 + it),
            # smaller loss weights avoids pushing too much to wrong direction, which can leads to strong interpenetration/object floating in the air
            'pinit': lambda cst, it: 250. ** 2 * cst / (1 + it), # even larger loss weight to prevent too much pose deviation
            # 'tinit': lambda cst, it: 10. ** 2 * cst / (1 + it),
            'tinit': lambda cst, it: 10. ** 2 * cst / (1 + it), # stronger t-init loss to prevent it move far away, leading to a distribution shift
            'normal': lambda cst, it: 2.5 ** 2 * cst / (1 + it), # weights for normal, adjust for batchfied version
            # 'normal': lambda cst, it: 2 ** 2 * cst / (1 + it), # even smaller weights
            'cdir': lambda cst, it: 5. ** 2 * cst / (1 + it),

        }
        return loss_weight

    def process_data_dict(self, data_dict):
        "convert list of indices to padded indices"
        obj_meshes: Meshes = data_dict['obj_meshes']
        obj_verts_padded = obj_meshes.verts_padded()
        bs, num_verts_obj = obj_verts_padded.shape[:2]  # (B, V, 3)
        num_verts_smpl = 6890

        verts_ind_obj_acc = torch.arange(bs, dtype=torch.long).to(self.device) * num_verts_obj
        verts_ind_smpl_acc = torch.arange(bs, dtype=torch.long).to(self.device) * num_verts_smpl

        bary_verts_ids_smpl, bary_verts_ids_obj = [], []
        cont_face_smpl, cont_face_obj = [], []
        for i in range(len(verts_ind_obj_acc)):
            if len(data_dict['bary_verts_ids_obj'][i]) == 0:
                print(f'no cotact for index {i}')
                assert len(data_dict['bary_verts_ids_smpl'][i]) == 0
                assert len(data_dict['cont_face_smpl'][i]) == 0
                assert len(data_dict['cont_face_obj'][i]) == 0
                continue
            # add to list
            bary_verts_ids_smpl.append(data_dict['bary_verts_ids_smpl'][i] + verts_ind_smpl_acc[i])
            bary_verts_ids_obj.append(data_dict['bary_verts_ids_obj'][i] + verts_ind_obj_acc[i])
            cont_face_obj.append(data_dict['cont_face_obj'][i] + verts_ind_obj_acc[i])
            cont_face_smpl.append(data_dict['cont_face_smpl'][i] + verts_ind_smpl_acc[i])
        cont_face_smpl = torch.cat(cont_face_smpl, 0)
        cont_face_obj = torch.cat(cont_face_obj, 0)
        bary_verts_ids_obj = torch.cat(bary_verts_ids_obj, 0)
        bary_verts_ids_smpl = torch.cat(bary_verts_ids_smpl, 0)

        # compute a weighting factor for each contact point
        num_contacts = [len(cont) for cont in data_dict['bary_verts_ids_obj']]
        weights, cont_verts_indices = [], []
        for i in range(bs):
            weight = torch.ones(num_contacts[i])/num_contacts[i]/bs
            weights.append(weight)
            cont_verts_indices.append(np.sum(num_contacts[:i+1])) # accumulated indices
        weights = torch.cat(weights).to(self.device) # (N,)

        bary_coords_smpl = torch.cat(data_dict['bary_coords_smpl'], 0).unsqueeze(-1)  # (N, 3, 1)
        bary_coords_obj = torch.cat(data_dict['bary_coords_obj'], 0).unsqueeze(-1)  # (N, 3, 1)

        cont_dir_orig = torch.cat(data_dict['cont_dir_orig'], 0).reshape(-1, 3, 1)

        data_dict['cont_dir_orig_packed'] = cont_dir_orig
        data_dict['bary_coords_smpl_packed'] = bary_coords_smpl
        data_dict['bary_coords_obj_packed'] = bary_coords_obj
        data_dict['bary_verts_ids_smpl_packed'] = bary_verts_ids_smpl
        data_dict['bary_verts_ids_obj_packed'] = bary_verts_ids_obj
        data_dict['cont_face_smpl_packed'] = cont_face_smpl
        data_dict['cont_face_obj_packed'] = cont_face_obj
        data_dict['per_point_weights'] = weights
        data_dict['cont_verts_indices'] = cont_verts_indices

        return data_dict

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-s', '--seqs_pattern', help='pattern to find interaction sequences to sample from')
        parser.add_argument('-fs', '--start', type=int, default=0)
        parser.add_argument('-fe', '--end', type=int, default=None)
        parser.add_argument('-obj', '--object', default='chairblack', type=str, help='object name of the original dataset')
        parser.add_argument('-sr', '--newshape_root', default="/mnt/d2/data/ShapenetV2/", help='root path to new object shape meshes')
        parser.add_argument('-scr', '--newshape_corr_root', default="/BS/xxie-2/static00/shapenet/",
                            help='root path to new object shape correspondence points')
        parser.add_argument('-cat', '--object_category', help='object category name for shapenet/objaverse/abo')
        parser.add_argument('-o', '--outfolder', default='outputs')
        parser.add_argument('-d', '--debug', default=False, action='store_true')

        parser.add_argument('-i', '--iterations', type=int, default=500)
        parser.add_argument('-tg', '--two_gender', default=False, action='store_true')
        parser.add_argument('-src', '--source', default='shapenet', choices=['shapenet', 'objaverse', 'abo'])

        # randomly select shape or not
        parser.add_argument('-random_shape', default=False, action='store_true')

        # new interaction optimization parameters
        parser.add_argument('-bs', '--batch_size', type=int, default=64)
        return parser

def main():
    from synz.mesh_paths import get_template_path, get_corr_mesh_file

    parser = BatchSynthesizer.get_parser()
    args = parser.parse_args()

    # For debug
    behave_params_root = '/scratch/inf0/user/xxie/behave-30fps'
    args.seqs_pattern = 'Date01_Sub01*chairblack*'
    smplh_root = '/BS/xxie2020/static00/mysmpl/smplh/'
    args.newshape_root = '/BS/databases24/ShapeNetV2-simplified'
    args.newshape_corr_root = '/BS/xxie-2/static00/shapenet/'
    args.object = 'chairblack'
    args.source = 'shapenet'
    args.newshape_category = 'chair'
    args.outfolder = f'outputs/test-{args.object}'
    args.batch_size = 16
    args.iterations = 500
    args.end = 64

    temp_path = get_template_path(args.object)
    # behave object points that have one to one corr to new shapes, this should align with the object mesh template above
    corr_mesh_file = get_corr_mesh_file(args.object)
    corr_mesh_file = osp.join('/BS/xxie-2/work/DualSDF2/', corr_mesh_file)
    assert args.object in args.seqs_pattern, f'the given object name does not compatible with the sequence pattern({args.seqs_pattern}) to sample from!'

    synzer = BatchSynthesizer(args.seqs_pattern,
                              behave_params_root,
                              smplh_root,
                              temp_path,
                              corr_mesh_file,
                              newshape_corr_root=args.newshape_corr_root,
                              newshape_root=args.newshape_root,
                              )
    synzer.synthesize(args)
    print('all done')



if __name__ == '__main__':
    import traceback
    try:
        main()
    except Exception as e:
        traceback.print_exc()