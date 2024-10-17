import torch
import numpy as np
import random

alphabet_mapper = {'<PAD>': 0}
PAD = alphabet_mapper['<PAD>']

def encode(str_indicies,max_length):
    if max_length == -1:
        local_max_length = max([len(x) for x in str_indicies])  # padding
    else:
        local_max_length = max_length
    

    str_indicies = list(str_indicies)
    nb = len(str_indicies)

    targets = torch.zeros(nb, local_max_length)
    targets[:, :] = PAD
    
    # import pdb;pdb.set_trace()
    # print(torch.Tensor(str_indicies[0]).shape)
    # targets[0,:len(str_indicies[0])]
    for i in range(nb):
        targets[i,:len(str_indicies[i])] = torch.Tensor(str_indicies[i])
    
    # text = targets.transpose(0, 1).contiguous()
    text = targets.long()
    mask = text == 0  # pad_mask
    # mask = mask.unsqueeze(1).expand(nb, mask.shape[1], mask.shape[1])
    
    return torch.LongTensor(text), mask

def set_seed(seeds):
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    np.random.seed(seeds)
    random.seed(seeds)