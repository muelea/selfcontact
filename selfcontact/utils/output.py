import numpy as np

def printlosses(loss_dict):
    lossstr = 'Losses |'
    for key, val in loss_dict.items():
        lossstr += '| {} {} '.format(key, np.round(val, 4))
    return lossstr