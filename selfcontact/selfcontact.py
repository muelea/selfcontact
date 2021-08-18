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
import numpy as np
import pickle
from .utils.mesh import batch_face_normals, \
                       batch_pairwise_dist, \
                       winding_numbers
from .body_segmentation import BatchBodySegment
from .utils.sparse import sparse_batch_mm

import os.path as osp

class SelfContact(nn.Module):
    def __init__(self,
        geodesics_path='',
        hd_operator_path='',
        point_vert_corres_path='',
        segments_folder='',
        faces_path='',
        essentials_folder=None,
        geothres=0.3,
        euclthres=0.02,
        model_type='smplx',
        test_segments=True,
        compute_hd=False,
        buffer_geodists=False,
    ):
        super().__init__()

        # contact thresholds
        self.model_type = model_type
        self.euclthres = euclthres
        self.geothres = geothres
        self.test_segments = test_segments
        self.compute_hd = compute_hd

        if essentials_folder is not None:
            geodesics_path = osp.join(essentials_folder, 'geodesics', 
                model_type, f'{model_type}_neutral_geodesic_dist.npy')
            hd_operator_path = osp.join(essentials_folder, 'hd_model', 
                model_type, f'{model_type}_neutral_hd_vert_regressor_sparse.npz')
            point_vert_corres_path = osp.join(essentials_folder, 'hd_model', 
                model_type, f'{model_type}_neutral_hd_sample_from_mesh_out.pkl')
            faces_path =  osp.join(essentials_folder, 'models_utils', 
                model_type, f'{model_type}_faces.npy')
            segments_folder = osp.join(essentials_folder, 'segments', 
                model_type)
            segments_bounds_path = f'{segments_folder}/{model_type}_segments_bounds.pkl'

        # create faces tensor
        faces = np.load(faces_path)
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

        # create extra vertex and faces to close back of the mouth to maske
        # the smplx mesh watertight.
        if self.model_type == 'smplx':
            inner_mouth_verts_path = f'{segments_folder}/smplx_inner_mouth_bounds.pkl'
            vert_ids_wt = np.array(pickle.load(open(inner_mouth_verts_path, 'rb')))
            self.register_buffer('vert_ids_wt', torch.from_numpy(vert_ids_wt))
            faces_wt = [[vert_ids_wt[i+1], vert_ids_wt[i],
                faces.max().item()+1] for i in range(len(vert_ids_wt)-1)]
            faces_wt = torch.tensor(np.array(faces_wt).astype(np.int64),
                dtype=torch.long)
            faces_wt = torch.cat((faces, faces_wt), 0)
            self.register_buffer('faces_wt', faces_wt)

        # geodesic distance mask
        if geodesics_path is not None:
            geodesicdists = torch.Tensor(np.load(geodesics_path))
            if buffer_geodists:
                self.register_buffer('geodesicdists', geodesicdists)
            geodistmask = geodesicdists >= self.geothres
            self.register_buffer('geomask', geodistmask)

        # create batch segmentation here
        if self.test_segments:
            sxseg = pickle.load(open(segments_bounds_path, 'rb'))
            self.segments = BatchBodySegment(
                [x for x in sxseg.keys()], faces, segments_folder, self.model_type
            )

        # load regressor to get high density mesh
        if self.compute_hd:
            hd_operator = np.load(hd_operator_path)
            hd_operator = torch.sparse.FloatTensor(
                torch.tensor(hd_operator['index_row_col']),
                torch.tensor(hd_operator['values']),
                torch.Size(hd_operator['size']))
            self.register_buffer('hd_operator',
                torch.tensor(hd_operator).float())

            with open(point_vert_corres_path, 'rb') as f:
                hd_geovec = pickle.load(f)['faces_vert_is_sampled_from']
            self.register_buffer('geovec',
                torch.tensor(hd_geovec))
            self.register_buffer('geovec_verts', self.faces[self.geovec][:,0])

    def triangles(self, vertices):
        # get triangles (close mouth for smplx)
        if self.model_type == 'smpl':
            triangles = vertices[:,self.faces,:]
        elif self.model_type == 'smplx':
            mouth_vert = torch.mean(vertices[:,self.vert_ids_wt,:], 1,
                        keepdim=True)
            vertices_mc = torch.cat((vertices, mouth_vert), 1)
            triangles = vertices_mc[:,self.faces_wt,:]
        return triangles

    def get_intersection_mask(self, vertices, triangles, test_segments=True):
        """
            compute status of vertex: inside, outside, or colliding
        """

        bs, nv, _ = vertices.shape

        # split because of memory into two chunks
        exterior = torch.zeros((bs, nv), device=vertices.device,
            dtype=torch.bool)
        exterior[:, :5000] = winding_numbers(vertices[:,:5000,:],
            triangles).le(0.99)
        exterior[:, 5000:] = winding_numbers(vertices[:,5000:,:],
            triangles).le(0.99)

        # check if intersections happen within segments
        if test_segments and not self.test_segments:
            assert('Segments not created. Create module with segments.')
            sys.exit()
        if test_segments and self.test_segments:
            for segm_name in self.segments.names:
                segm_vids = self.segments.segmentation[segm_name].segment_vidx
                for bidx in range(bs):
                    if (exterior[bidx, segm_vids] == 0).sum() > 0:
                        segm_verts = vertices[bidx, segm_vids, :].unsqueeze(0)
                        segm_ext = self.segments.segmentation[segm_name] \
                            .has_self_isect_points(
                                segm_verts.detach(),
                                triangles[bidx].unsqueeze(0)
                        )
                        mask = ~segm_ext[bidx]
                        segm_idxs = torch.masked_select(segm_vids, mask)
                        true_tensor = torch.ones(segm_idxs.shape, device=segm_idxs.device, dtype=torch.bool)
                        exterior[bidx, segm_idxs] = true_tensor

        return exterior

    def get_hd_intersection_mask(self, points, triangles,
        faces_ioc_idx, hd_verts_ioc_idx, test_segments=False):
        """
            compute status of vertex: inside, outside, or colliding
        """
        bs, np, _ = points.shape
        assert bs == 1, 'HD points intersections only work with batch size 1.'

        # split because of memory into two chunks
        exterior = torch.zeros((bs, np), device=points.device,
            dtype=torch.bool)
        exterior[:, :6000] = winding_numbers(points[:,:6000,:],
            triangles).le(0.99)
        exterior[:, 6000:] = winding_numbers(points[:,6000:,:],
            triangles).le(0.99)

        return exterior

    def get_pairwise_dists(self, verts1, verts2, squared=False):
        """
            compute pairwise distance between vertices
        """
        v2v = batch_pairwise_dist(verts1, verts2, squared=squared)
        return v2v

    def segment_vertices(self, vertices, compute_hd=False, test_segments=True):
        """
            get self-intersecting vertices and pairwise distance
        """
        bs = vertices.shape[0]

        triangles = self.triangles(vertices.detach())

        # get inside / outside segmentation
        exterior = self.get_intersection_mask(
                vertices.detach(),
                triangles.detach(),
                test_segments
        )

        # get pairwise distances of vertices
        v2v = self.get_pairwise_dists(vertices, vertices, squared=False)
        v2v_mask = v2v.detach().clone()
        inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device)
        v2v_mask[:, ~self.geomask] = inf_tensor
        _, v2v_min_index = torch.min(v2v_mask, dim=1)
        v2v_min = torch.gather(v2v, dim=2,
            index=v2v_min_index.view(bs,-1,1)).squeeze(-1)
        incontact = v2v_min < self.euclthres

        hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact = None, None, None, None
        if compute_hd:
            hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact = \
                self.segment_hd_points(
                    vertices, v2v_min, incontact, exterior, test_segments)

        v2v_out = (v2v_min, incontact, exterior)
        hd_v2v_out = (hd_v2v_min, hd_exterior, hd_points, hd_faces_in_contact)

        return v2v_out, hd_v2v_out

    def segment_vertices_scopti(self, vertices, test_segments=True):
        """
            get self-intersecting vertices and pairwise distance 
            for self-contact optimization. This version is determinisic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        # get pairwise distances of vertices
        v2v = vertices.squeeze().unsqueeze(1).expand(nv, nv, 3) - \
                vertices.squeeze().unsqueeze(0).expand(nv, nv, 3)
        v2v = torch.norm(v2v, dim=2).unsqueeze(0)

        with torch.no_grad():
            triangles = self.triangles(vertices.detach())

            # get inside / outside segmentation
            exterior = self.get_intersection_mask(
                    vertices.detach(),
                    triangles.detach(),
                    test_segments
            )

            v2v_mask = v2v.detach().clone()
            #v2v_mask[:, ~self.geomask] = float('inf')
            inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device)
            v2v_mask[:, ~self.geomask] = inf_tensor
            _, v2v_min_index = torch.min(v2v_mask, dim=1)

        #v2v_min = torch.gather(v2v, dim=2,
        #    index=v2v_min_index.view(bs,-1,1)).squeeze(-1)
        v2v_min = v2v[:, np.arange(nv), v2v_min_index[0]]

        return (v2v_min, v2v_min_index, exterior)

    def segment_points_scopti(self, points, vertices):
        """
            get self-intersecting points (vertices on extremities) and pairwise distance
            for self-contact optimization. This version is determinisic.
        """
        bs, nv, _ = vertices.shape
        if bs > 1:
            sys.exit('Please use batch size one or set use_pytorch_norm=False')

        v2v = vertices.squeeze().unsqueeze(1).expand(nv, nv, 3) - \
                vertices.squeeze().unsqueeze(0).expand(nv, nv, 3)
        v2v = torch.norm(v2v, dim=2).unsqueeze(0)

        # find closest vertex in contact
        with torch.no_grad():
            triangles = self.triangles(vertices.detach())

            # get inside / outside segmentation
            exterior = self.get_intersection_mask(
                    vertices=points.detach(),
                    triangles=triangles.detach(),
                    test_segments=False
            )

            v2v_mask = v2v.detach().clone()
            #v2v_mask[:, ~self.geomask] = float('inf')
            inf_tensor = float('inf') * torch.ones((1,(~self.geomask).sum().item()), device=v2v.device)
            v2v_mask[:, ~self.geomask] = inf_tensor
            _, v2v_min_index = torch.min(v2v_mask, dim=1)

        # first version is better, but not deterministic
        #v2v_min = torch.gather(v2v, dim=2,
        #    index=v2v_min_index.view(bs,-1,1)).squeeze(-1)
        v2v_min = v2v[:, np.arange(nv), v2v_min_index[0]]
        
        return (v2v_min, v2v_min_index, exterior)

    def segment_hd_points(self, vertices, v2v_min, incontact, exterior, test_segments=True):
        """
            compute hd points from vertices and compute their distance
            and inside / outside segmentation
        """
        bs, nv, _ = vertices.shape

        # select all vertices that are inside or in contact
        verts_ioc = incontact | ~exterior
        verts_ioc_idxs = torch.nonzero(verts_ioc)
        verts_ioc_idx = torch.nonzero(verts_ioc[0]).view(-1)

        # get hd points for inside or in contact vertices

        # get hd points for inside or in contact vertices
        hd_v2v_mins = []
        hd_exteriors = []
        hd_points = []
        hd_faces_in_contacts = []
        for idx in range(bs):
            verts_ioc_idx = torch.nonzero(verts_ioc[idx]).view(-1)
            exp1 = verts_ioc_idx.expand(self.faces.flatten().shape[0], -1)
            exp2 = self.faces.flatten().unsqueeze(-1).expand(-1, verts_ioc_idx.shape[0])
            nzv = (exp1 == exp2).any(1).reshape(-1, 3).any(1)
            faces_ioc_idx = torch.nonzero(nzv).view(-1)
            hd_verts_ioc_idx = (self.geovec.unsqueeze(-1) == \
                faces_ioc_idx.expand(self.geovec.shape[0], -1)).any(1)


            # check is points were samples from the vertices in contact
            if hd_verts_ioc_idx.sum() > 0:
                hd_verts_ioc = sparse_batch_mm(self.hd_operator, vertices[[idx]])[:,hd_verts_ioc_idx,:]
                triangles = self.triangles(vertices[[idx]])
                face_normals = batch_face_normals(triangles)[0]
                with torch.no_grad():
                    hd_v2v = self.get_pairwise_dists(hd_verts_ioc, hd_verts_ioc, squared=True)
                    geom_idx = self.geovec_verts[hd_verts_ioc_idx]
                    hd_geo = self.geomask[geom_idx,:][:,geom_idx]
                    #hd_v2v[:, ~hd_geo] = float('inf')
                    inf_tensor = float('inf') * torch.ones((1,(~hd_geo).sum().item()), device=hd_v2v.device)
                    hd_v2v[:, ~self.geomask] = inf_tensor
                    hd_v2v_min, hd_v2v_min_idx = torch.min(hd_v2v, dim=1)

                    # add little offset to those vertices for in/ex computation
                    faces_ioc_idx = self.geovec[hd_verts_ioc_idx]
                    hd_verts_ioc_offset = hd_verts_ioc + \
                        0.001 * face_normals[faces_ioc_idx, :].unsqueeze(0)

                    # test if hd point is in- or outside
                    hd_exterior = self.get_hd_intersection_mask(
                        hd_verts_ioc_offset.detach(),
                        triangles.detach(),
                        faces_ioc_idx=faces_ioc_idx,
                        hd_verts_ioc_idx=hd_verts_ioc_idx,
                        test_segments=False,
                    )[0]

                    hd_close_faces = torch.vstack((faces_ioc_idx, faces_ioc_idx[hd_v2v_min_idx[0]]))
                    hd_verts_in_close_contact = hd_v2v_min < 0.005**2
                    hd_faces_in_contact = hd_close_faces[:, hd_verts_in_close_contact[0]]

                hd_v2v_min = torch.norm(hd_verts_ioc[0] - hd_verts_ioc[0, hd_v2v_min_idx, :], dim=2)[0]
                hd_v2v_mins += [hd_v2v_min]
                hd_faces_in_contacts += [hd_faces_in_contact]
                hd_exteriors += [hd_exterior]
                hd_points += [hd_verts_ioc_offset]
            else:
                hd_v2v_mins += [None]
                hd_exteriors += [None]
                hd_points += [None]
                hd_faces_in_contacts += [None]

        return hd_v2v_mins, hd_exteriors, hd_points, hd_faces_in_contacts



class SelfContactSmall(nn.Module):
    def __init__(self,
        essentials_folder,
        geothres=0.3,
        euclthres=0.02,
        model_type='smplx',
    ):
        super().__init__()

        self.model_type = model_type

        # contact thresholds
        self.euclthres = euclthres
        self.geothres = geothres

        # geodesic distance mask
        geodesics_path = osp.join(essentials_folder, 'geodesics', model_type, f'{model_type}_neutral_geodesic_dist.npy')
        geodesicdists = torch.Tensor(np.load(geodesics_path))
        geodistmask = geodesicdists >= self.geothres
        self.register_buffer('geomask', geodistmask)

        # create faces tensor
        faces_path =  osp.join(essentials_folder, 'models_utils', model_type, f'{model_type}_faces.npy')
        faces = np.load(faces_path)
        if type(faces) is not torch.Tensor:
            faces = torch.tensor(faces.astype(np.int64), dtype=torch.long)
        self.register_buffer('faces', faces)

    def get_pairwise_dists(self, verts1, verts2, squared=False):
        """
            compute pairwise distance between vertices
        """
        v2v = batch_pairwise_dist(verts1, verts2, squared=squared)

        return v2v
  
    def pairwise_selfcontact_for_verts(self, vertices):
            """
            Returns tensor of vertex pairs that are in contact. If you have a batch of vertices,
            the number of vertices returned per mesh can be different. To get verts in contact 
            for batch_index_x use:
            batch_x_verts_in_contact = contact[torch.where(in_contact_batch_idx == batch_index_x)[0], :]
            """

            # get pairwise distances of vertices
            v2v = self.get_pairwise_dists(vertices, vertices, squared=True)

            # find closes vertex in contact
            v2v[:, ~ self.geomask] = float('inf')
            v2v_min, v2v_min_index = torch.min(v2v, dim=1)
            in_contact_batch_idx, in_contact_idx1 = torch.where(v2v_min < self.euclthres**2)
            in_contact_idx2 = v2v_min_index[in_contact_batch_idx, in_contact_idx1]
            contact = torch.vstack((in_contact_idx1, in_contact_idx2)).view(-1,2,1)

            return in_contact_batch_idx, contact


    def verts_in_contact(self, vertices, return_idx=False):

            # get pairwise distances of vertices
            v2v = self.get_pairwise_dists(vertices, vertices, squared=True)

            # mask v2v with eucledean and geodesic dsitance
            euclmask = v2v < self.euclthres**2
            mask = euclmask * self.geomask

            # find closes vertex in contact
            in_contact = mask.sum(1) > 0

            if return_idx:
                in_contact = torch.where(in_contact)

            return in_contact
