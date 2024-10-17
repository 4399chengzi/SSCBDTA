import torch
from torch.utils.data import Dataset
import sys
sys.path.append('./')
import DataHelper as DH
from label_util import encode
# import DataHelper as DH

class DistCollateFn(object):
    '''
    fix bug when len(data) do not be divided by batch_size, on condition of distributed validation
    avoid error when some gpu assigned zero samples
    '''

    def __call__(self, batch):

        # if batch_size == 0:
        #     return dict(batch_size = batch_size, images = None, labels = None)

        # if self.training:
        drug_indicies, target_indicies , affinity = zip(*batch)
        # image_batch_tensor = torch.stack(images, dim = 0).float()
        # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
        return dict(
                    drug_indicies = drug_indicies,
                    target_indicies = target_indicies,
                    affinity = affinity)


class Davis_kiba_data(Dataset):
    def __init__(self,root_path, data_choice ,drugSeq_maxlenKB=100, proSeq_maxlenKB=1000):
        super(Davis_kiba_data, self).__init__()

        # self.alphabet_mapper = {'<EOS>': 1, '<SOS>': 2, '<PAD>': 0, '<UNK>': 3}
        # self.PAD = self.alphabet_mapper['<PAD>']

        self.root_path = root_path
        self.drugSeq_maxlenKB = drugSeq_maxlenKB
        self.proSeq_maxlenKB = proSeq_maxlenKB
        assert data_choice in ['davis', 'kiba']

        drug, target, affinity = DH.LoadData(self.root_path, logspance_trans=False)
        drug_seqs, target_seqs, affiMatrix = DH.GetSamples(data_choice, drug, target, affinity)
        labeled_drugs, labeled_targets = DH.LabelDT(drug_seqs, target_seqs, drugSeq_maxlenKB, proSeq_maxlenKB)
        self.labeledDrugs_shuttle, self.labeledTargets_shuttle, self.affiMatrix_shuttle \
                                   = DH.shuttle(labeled_drugs, labeled_targets, affiMatrix)
    
    def __len__(self):
        return self.labeledDrugs_shuttle.shape[0]
    
    def __getitem__(self, index):
        # nb = 1 # TODO
        # drug_indicies = torch.zeros(nb, self.drugSeq_maxlenKB)
        # drug_indicies[:, :] = self.PAD
        # target_indicies = torch.zeros(nb, self.proSeq_maxlenKB)
        # target_indicies[:, :] = self.PAD

        # drug_length = len(self.labeledDrugs_shuttle[index])
        drug_indicies = self.labeledDrugs_shuttle[index]
        # target_length = len(self.labeledTargets_shuttle[index])
        target_indicies = self.labeledTargets_shuttle[index]
        
        affinity = self.affiMatrix_shuttle[index]

        return drug_indicies, target_indicies , affinity


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    import torch.nn as nn
    sys.path.append('../')
    from label_util import encode
    
    root_path = '/home/wwyu/data/data_davis/'
    davis = Davis_kiba_data(root_path, 'davis')
    data_loader = DataLoader(davis, batch_size=1, collate_fn=DistCollateFn(), shuffle=False)
    
    # import pdb;pdb.set_trace()

    for i, dic in enumerate(data_loader):
        pad_drug, drug_mask = encode(dic['drug_indicies'], -1)
        pad_tar, tar_mask = encode(dic['target_indicies'], -1)
        
        print(pad_drug[0])
        print(drug_mask[0], drug_mask.shape)
        break