# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from selfcontact import SelfContact
import torch 
import trimesh 
import os.path as osp
import numpy as np
import argparse
import os 
import glob
import smplx
import yaml
from selfcontact.losses import SelfContactOptiLoss
from selfcontact.fitting import SelfContactOpti
from selfcontact.utils.parse import DotConfig

extremities_nums = [1,2,4,5,7,8,10,11,16,17,18,19,
    20,21,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
    39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]


def get_bodymodels(
        model_path, 
        model_type, 
        device, 
        batch_size=1, 
        num_pca_comps=12
    ):

    models = {}

    # model parameters for self-contact optimization
    model_params = dict(
        batch_size=batch_size,
        model_type=model_type,
        create_body_pose=True,
        create_transl=False,
        create_betas=False,
        create_global_orient=False,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        use_pca=True,
        num_pca_comps=num_pca_comps,
        return_full_pose=True
    )
    
    # create smplx model per gender
    for gender in ['male', 'female', 'neutral']:
        models[gender] = smplx.create(
            model_path=model_path,
            gender=gender,
            **model_params
        ).to(device)
    
    return models

def main(cfg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # process arguments
    INPUT_FN = cfg.input_folder
    OUTPUT_DIR = cfg.output_folder
    HCP_PATH = osp.join(cfg.essentials_folder, 'hand_on_body_prior/smplx/smplx_handonbody_prior.pkl')
    SEGMENTATION_PATH = osp.join(cfg.essentials_folder, 'smplify/smplx_segmentation_id.npy')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # extrimities vertex IDs
    smpl_bone_vert = np.load(SEGMENTATION_PATH)
    extremities = np.where(np.isin(smpl_bone_vert, extremities_nums))[0]
    extremities = torch.tensor(extremities)

    models = get_bodymodels(
        model_path=cfg.model_folder, 
        model_type=cfg.body_model.model_type, 
        device=device,
        batch_size=cfg.batch_size, 
        num_pca_comps=cfg.body_model.num_pca_comps
    )

    sc_module = SelfContact( 
        essentials_folder=cfg.essentials_folder,
        geothres=cfg.contact.geodesic_thres, 
        euclthres=cfg.contact.euclidean_thres, 
        model_type=cfg.body_model.model_type,
        test_segments=False,
        compute_hd=False,
        buffer_geodists=True,
    ).to(device)

    criterion = SelfContactOptiLoss( 
        contact_module=sc_module,
        inside_loss_weight=cfg.loss.inside_weight,
        outside_loss_weight=cfg.loss.outside_weight,
        contact_loss_weight=cfg.loss.contact_weight,
        hand_contact_prior_weight=cfg.loss.hand_contact_prior_weight,
        hand_contact_prior_path=HCP_PATH,
        downsample=extremities,
        device=device
    )

    scopti = SelfContactOpti(
        loss=criterion,
        optimizer_name=cfg.optimizer.name,
        optimizer_lr=cfg.optimizer.learning_rate,
        max_iters=cfg.optimizer.max_iters,
    )
   
    # read files 
    npz_files = glob.glob(osp.join(INPUT_FN, '*.npz'))

    for npz_idx, npz_file in enumerate(npz_files):

        print('Processing: ', npz_file)

        data = np.load(npz_file)
        body_pose = torch.Tensor(data['body_pose'][3:66]).unsqueeze(0)

        # most AMASS meshes don't have hand poses, so we don't use them here.
        #left_hand_pose = torch.Tensor(data['body_pose'][75:81]).unsqueeze(0)
        #right_hand_pose = torch.Tensor(data['body_pose'][81:]).unsqueeze(0)

        global_orient = torch.zeros(1,3) if 'global_orient' not in data.keys() \
            else data['global_orient']
        betas = torch.from_numpy(data['betas']).unsqueeze(0).float()
        gender = data['gender'][0].decode("utf-8")

        params = dict(
            betas = betas.to(device),
            global_orient = global_orient.to(device),
            body_pose = body_pose.to(device),
            #left_hand_pose = left_hand_pose,
            #right_hand_pose = right_hand_pose
        )

        model = models[gender]
        model.reset_params(**params)

        body = scopti.run(model, params, npz_idx)

        mesh = trimesh.Trimesh(body.vertices[0].detach().cpu().numpy(), model.faces)
        mesh.export(osp.join(OUTPUT_DIR, f'{npz_idx}_output.obj'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='selfcontact/tutorial/configs/selfcontact_optimization_config.yaml')
    parser.add_argument('--essentials_folder', required=True, 
        help='folder with essential data. Check Readme for download dir.')
    parser.add_argument('--output_folder', required=True, 
        help='folder where the example obj files are written to')
    parser.add_argument('--model_folder', required=True, 
        help='folder where the body models are saved.')
    parser.add_argument('--input_folder', default='', 
        help='Input filename to be processed. Expects an npz file with the model parameters.')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        cfg = DotConfig(
            input_dict=yaml.safe_load(stream)
        )
    
    cfg.update(vars(args))

    main(cfg)