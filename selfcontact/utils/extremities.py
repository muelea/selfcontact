import numpy as np
import torch

smplx_extremities_nums = [1,2,4,5,7,8,10,11,16,17,18,19,
    20,21,25,26,27,28,29,30,31,32,33,34,35,36,37,38,
    39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
]
smplx_inner_toes = [5779, 5795, 5796, 5797, 5798, 5803, 5804, 5814, 5815, 
    5819, 5820, 5823, 5824, 5828, 5829, 5840, 5841, 5852, 5854, 5862, 5864, 
    8472, 8473, 8489, 8490, 8491, 8492, 8497, 8498, 8508, 8509, 8513, 8514,
    8517, 8518, 8522, 8523, 8531, 8532, 8533, 8534, 8535, 8536, 8546, 8548, 
    8556, 8558, 8565
]

def get_extremities(segmentation_path, include_toes=False):

    # extrimities vertex IDs
    smpl_bone_vert = np.load(segmentation_path)

    extremities = np.where(np.isin(smpl_bone_vert, smplx_extremities_nums))[0]
    # ignore inner toe vertices by default
    if not include_toes:
        extremities = extremities[~np.isin(extremities, smplx_inner_toes)]
    extremities = torch.tensor(extremities)

    return extremities