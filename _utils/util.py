import torch
import numpy as np
import random
from torch.nn.parameter import Parameter

# Set seed for reproduction
def fix_random_seed_for_reproduce(torch_seed=2, np_seed=3):
    # fix random seeds for reproducibility,
    random.seed(np_seed)
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)  # for current gpu
    torch.cuda.manual_seed_all(torch_seed)  # for all gpu
    torch.backends.cudnn.benchmark = False  # if benchmark=True, speed up training, and deterministic will set be False
    torch.backends.cudnn.deterministic = True  # which can slow down training considerably
