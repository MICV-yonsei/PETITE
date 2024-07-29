import random
import numpy as np 
import torch

def fix_seed(seed_num=0):
    """Fix seed before load dataset and initialize model"""
    # python
    random.seed(seed_num)

    # numpy
    np.random.seed(seed_num)

    # pytorch
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)  

    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
