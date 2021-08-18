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

import torch 
import trimesh
import os

import numpy as np

from ..utils.output import printlosses

class SelfContactOpti():
    def __init__(
        self,
        loss,
        optimizer_name='adam',
        optimizer_lr_body=0.01,
        optimizer_lr_hands=0.01,
        max_iters=100,
        loss_thres=1e-5,
        patience=5,

    ):
        super().__init__()

        # create optimizer
        self.optimizer_name =  optimizer_name 
        self.optimizer_lr_body = optimizer_lr_body
        self.optimizer_lr_hands = optimizer_lr_hands
        self.max_iters = max_iters
        self.loss_thres = loss_thres
        self.patience = patience

        # self-contact optimization loss
        self.loss = loss

    def get_optimizer(self, model):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam([
                {'params': model.body_pose, 'lr': self.optimizer_lr_body},
                {'params': [model.left_hand_pose, model.right_hand_pose], 'lr': self.optimizer_lr_hands},
            ])
        return optimizer

    def run(self, body_model, params):

        # create optimizer 
        optimizer = self.get_optimizer(body_model)

        # configure loss with initial mesh
        self.loss.configure(body_model, params)

        # initial body
        body = body_model(
            get_skin=True,
            global_orient=params['global_orient'],
            betas=params['betas']
        )

        # Initialize optimization
        step = 0
        criterion_loss = True
        loss_old = 0
        count_loss = 0

        # run optimization
        while step < self.max_iters and criterion_loss:

            optimizer.zero_grad()

            # get current model
            body = body_model(
                get_skin=True,
                global_orient=params['global_orient'],
                betas=params['betas']
            )

            #mesh = trimesh.Trimesh(body.vertices[0].detach().cpu().numpy(), body_model.faces)
            #mesh.export(os.path.join('/is/cluster/lmueller2/outdebug/aaa_blaaaa/new2', f'{step:02f}output.obj'))

            # compute loss
            total_loss, loss_dict = self.loss(body)
            print(step, printlosses(loss_dict))

            # =========== stop criterion based on loss ===========
            with torch.no_grad():
                count_loss = count_loss + 1 if abs(total_loss - loss_old) < self.loss_thres else 0
                if count_loss >= self.patience:
                    criterion_loss = False

                loss_old = total_loss
                step += 1

            # back prop
            total_loss.backward(retain_graph=False)
            optimizer.step()

        return body

