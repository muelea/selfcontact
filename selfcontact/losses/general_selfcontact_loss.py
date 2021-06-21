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

import sys
import torch
import torch.nn as nn
from ..utils.mesh import batch_face_normals

class SelfContactLoss(nn.Module):
    def __init__(self,
        contact_module,
        inside_loss_weight=1.0,
        outside_loss_weight=1.0,
        contact_loss_weight=2.5,
        alpha1=0.005,
        alpha2=0.005,
        beta1=1.0,
        beta2=0.04,
        align_faces=True,
        use_hd = True,
        test_segments=True,
        device='cuda',
        model_type='smplx',
    ):
        super().__init__()

        self.device = device
        self.model_type = model_type

        # loss specifications
        self.use_hd = use_hd
        self.test_segments = test_segments
        self.align_faces = align_faces

        # push, pull, and contact weights
        self.inside_w = inside_loss_weight
        self.contact_w = contact_loss_weight
        self.outside_w = outside_loss_weight

        # hyper params
        self.a1 = alpha1
        self.a2 = alpha2
        self.b1 = beta1
        self.b2 = beta2

        # self contact module
        self.cm = contact_module

    def forward(self, vertices):
        """
            Loss that brings surfaces that are close into contact and
            resolves intersections.
        """
        bs = vertices.shape[0]

        v2v_pull = torch.zeros(bs, device=vertices.device)
        v2v_push = torch.zeros(bs, device=vertices.device)
        contactloss = torch.tensor(0, device=vertices.device)

        # if use_hd v2v and inside / outside segmentation and v2v is computed
        # on original model vertices else on hd surface points
        (verts_v2v_min, verts_incontact, verts_exterior), \
        (hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact) \
        = self.cm.segment_vertices(
            vertices,
            self.use_hd,
            test_segments=self.test_segments
        )

        if self.align_faces:
            triangles = self.cm.triangles(vertices)
            face_normals = batch_face_normals(triangles)

        for idx in range(bs):
            # select the correct set of vertices
            if self.use_hd:
                v2v_min, exterior = hd_v2v_min[idx], hd_exterior[idx]
            else:
                # select vertices that are inside or in contact
                v2v_min = verts_v2v_min[idx]
                exterior = verts_exterior[idx]

            if exterior is not None:
                # apply contact loss to vertices in contact
                if exterior.sum() > 0:
                    v2v_pull[idx] = self.contact_w * (self.a1 * \
                        torch.tanh(v2v_min[exterior] / self.a2)**2).sum()

                # apply contact loss to inside vertices
                if (~exterior).sum() > 0:
                    v2v_push[idx] = self.inside_w * (self.b1 * \
                        torch.tanh(v2v_min[~exterior] / self.b2)**2).sum()

            # now align faces that are close
            # dot product should be -1 for faces in contact
            face_angle_loss = torch.zeros(bs, device=vertices.device)
            if self.align_faces:
                if self.use_hd:
                    if hd_faces_in_contact[idx] is not None:
                        hd_fn_in_contact = face_normals[idx][hd_faces_in_contact[idx]]
                        dotprod_normals = 1 + (hd_fn_in_contact[0] * hd_fn_in_contact[1]).sum(1)
                        face_angle_loss[idx] = dotprod_normals.sum()
                else:
                    sys.exit('You can only align vertices when use_hd=True')


        # compute contact loss
        contactloss = v2v_pull + v2v_push

        return contactloss, face_angle_loss
