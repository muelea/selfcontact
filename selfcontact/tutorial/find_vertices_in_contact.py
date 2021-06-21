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

def save_mesh(mesh, in_contact, path='mesh_out.obj'):

    in_contact_np = in_contact.cpu().numpy()

    # color vertices inside red, outside green
    color = np.array(mesh.visual.vertex_colors)
    color[~in_contact_np[0], :] = [233, 233, 233, 255]
    color[in_contact_np[0], :] = [0, 0, 255, 255]
    mesh.visual.vertex_colors = color

    # export mesh
    mesh.export(path)

def main(args):

    # process arguments
    ESSENTIALS_DIR = args.essentials_folder    
    MODEL_TYPE = args.model_type
    DEVICE = args.device
    OBJ_FILE = args.obj_file
    OUTPUT_DIR = args.output_folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if OBJ_FILE == '':
        MESH_FN = 'heavy_lifting'
        OBJ_FILE = osp.join(ESSENTIALS_DIR, 'example_meshes/selfcontact', MESH_FN+'.obj')
    else:
        MESH_FN = osp.basename(OBJ_FILE).replace('.obj', '')

    sc_module = SelfContact( 
        essentials_folder=ESSENTIALS_DIR,
        geothres=0.3, 
        euclthres=0.02, 
        model_type=MODEL_TYPE,
        test_segments=True,
        compute_hd=False
    )

    mesh = trimesh.load(OBJ_FILE, process=False)
    vertices = torch.from_numpy(mesh.vertices) \
                    .unsqueeze(0) \
                    .to(DEVICE) \
                    .float()


    # Segment mesh into inside and outside vertices
    (verts_v2v_min, verts_incontact, verts_exterior), _ \
    = sc_module.segment_vertices(
        vertices,
        compute_hd=False,
        test_segments=False)

    save_mesh(mesh, verts_incontact, 
        osp.join(OUTPUT_DIR, f'{MESH_FN}.obj'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--essentials_folder', required=True, 
        help='folder with essential data. Check Readme for download dir.')
    parser.add_argument('--output_folder', required=True, 
        help='folder where the example obj files are written to')
    parser.add_argument('--model_type', default='smplx', 
        choices=['smpl', 'smplx'], help='body model to use')
    parser.add_argument('--device', default='cuda', help='torch device.')
    parser.add_argument('--obj_file', default='', 
        help='Input obj file to be processed. If no obj file is passed, a default obj from essentials will be selected.')

    args = parser.parse_args()

    main(args)