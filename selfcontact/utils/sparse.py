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

def sparse_batch_mm(m1, m2):
    """
    https://github.com/pytorch/pytorch/issues/14489

    m1: sparse matrix of size N x M
    m2: dense matrix of size B x M x K
    returns m1@m2 matrix of size B x N x K
    """

    batch_size = m2.shape[0]
    # stack m2 into columns: (B x N x K) -> (N, B, K) -> (N, B * K)
    m2_stack = m2.transpose(0, 1).reshape(m1.shape[1], -1)
    result = m1.mm(m2_stack).reshape(m1.shape[0], batch_size, -1) \
               .transpose(1, 0)
    return result
