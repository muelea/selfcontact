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
import smplx
import yaml
import pickle
from selfcontact.losses import SelfContactOptiLoss
from selfcontact.fitting import SelfContactOpti
from selfcontact.utils.parse import DotConfig
from selfcontact.utils.body_models import get_bodymodels
from selfcontact.utils.extremities import get_extremities

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True

# code contains non-deterministic parts, that may lead to
# different results when running the same script again. To
# run the determinisitc version use the following lines:
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)

def main(cfg):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # process arguments
    OUTPUT_DIR = cfg.output_folder
    HCP_PATH = osp.join(cfg.essentials_folder, 'hand_on_body_prior/smplx/smplx_handonbody_prior.pkl')
    SEGMENTATION_PATH = osp.join(cfg.essentials_folder, 'models_utils/smplx/smplx_segmentation_id.npy')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_file = osp.join(cfg.essentials_folder, f'example_poses/pose1.npz') \
        if cfg.input_file == '' else cfg.input_file
    print('Processing: ', npz_file)

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
        test_segments=cfg.contact.test_segments,
        compute_hd=False,
        buffer_geodists=True,
    ).to(device)

    extremities = get_extremities(
        SEGMENTATION_PATH, 
        cfg.contact.test_segments
    )
    criterion = SelfContactOptiLoss( 
        contact_module=sc_module,
        inside_loss_weight=cfg.loss.inside_weight,
        outside_loss_weight=cfg.loss.outside_weight,
        contact_loss_weight=cfg.loss.contact_weight,
        hand_contact_prior_weight=cfg.loss.hand_contact_prior_weight,
        pose_prior_weight=cfg.loss.pose_prior_weight,
        hand_pose_prior_weight=cfg.loss.hand_pose_prior_weight,
        angle_weight=cfg.loss.angle_weight,
        hand_contact_prior_path=HCP_PATH,
        downsample=extremities,
        use_hd=False,
        test_segments=cfg.contact.test_segments,
        device=device
    )

    scopti = SelfContactOpti(
        loss=criterion,
        optimizer_name=cfg.optimizer.name,
        optimizer_lr_body=cfg.optimizer.learning_rate_body,
        optimizer_lr_hands=cfg.optimizer.learning_rate_hands,
        max_iters=cfg.optimizer.max_iters,
    )
   
    # load data
    data = np.load(npz_file)
    gender = data['gender'][0].decode("utf-8")
    betas = torch.from_numpy(data['betas']).unsqueeze(0).float()
    global_orient = torch.zeros(1,3) if 'global_orient' not in data.keys() \
        else data['global_orient']
    body_pose = torch.Tensor(data['body_pose'][3:66]).unsqueeze(0)
    # most AMASS meshes don't have hand poses, so we don't use them here.
    #left_hand_pose = torch.Tensor(data['body_pose'][75:81]).unsqueeze(0)
    #right_hand_pose = torch.Tensor(data['body_pose'][81:]).unsqueeze(0)

    params = dict(
        betas = betas.to(device),
        global_orient = global_orient.to(device),
        body_pose = body_pose.to(device),
        #left_hand_pose = left_hand_pose,
        #right_hand_pose = right_hand_pose
    )

    model = models[gender]
    model.reset_params(**params)

    body = scopti.run(model, params)

    mesh = trimesh.Trimesh(body.vertices[0].detach().cpu().numpy(), model.faces)
    mesh.export(osp.join(OUTPUT_DIR, f'output.obj'))

    out_dict = {}
    for key, val in body.items():
        if val is not None:
            out_dict[key] = val.detach().cpu().numpy()
    with open(osp.join(OUTPUT_DIR, f'output.pkl'), 'wb') as f:
        pickle.dump(out_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', 
        default='selfcontact/tutorial/configs/selfcontact_optimization_config_orig.yaml')
    parser.add_argument('--essentials_folder', required=True, 
        help='folder with essential data. Check Readme for download dir.')
    parser.add_argument('--output_folder', required=True, 
        help='folder where the example obj files are written to')
    parser.add_argument('--model_folder', required=True, 
        help='folder where the body models are saved.')
    parser.add_argument('--input_file', default='', 
        help='Input filename to be processed. Expects an npz file with the model parameters.')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        cfg = DotConfig(
            input_dict=yaml.safe_load(stream)
        )
    
    cfg.update(vars(args))

    main(cfg)