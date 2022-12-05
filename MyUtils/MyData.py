import numpy as np
from torch.utils.data import Dataset

class HumanData(Dataset):
    def __init__(self,url='HumanByStr',mode='train'):
        '''items represents the list of training elements (drug, protein)'''
        super(HumanData, self).__init__()
        self.data=np.load('{}/{}_IndexVec_data.npy'.format(url,mode),allow_pickle=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        drug_indexVec, protein_indexVec, label=self.data[index]
        return drug_indexVec, protein_indexVec, label

class CelegansData(Dataset):
    def __init__(self,url='CelegansByStr',mode='train'):
        '''items represents the list of training elements (drug, protein) '''
        super(CelegansData, self).__init__()
        self.data=np.load('{}/{}_IndexVec_data.npy'.format(url,mode),allow_pickle=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        drug_indexVec, protein_indexVec, label=self.data[index]
        return drug_indexVec, protein_indexVec, label