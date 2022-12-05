import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    # Obtaining encoding vector according to protein sequence
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X

def getMF(smiles):
    # get Morgan Fingerprint By RDKit
    m1=Chem.MolFromSmiles(smiles)
    fg=AllChem.GetMorganFingerprintAsBitVect(m1,2)
    return fg

def getMF_Index_Vector(Fingerprint):
    # get the indexing Vector by Morgan Fingerprint
    return list(Fingerprint.GetOnBits())

def get_Smiles_Index_Vector(smiles):
    # get the indexing Vector of smiles sequence by RDKit
    MF=getMF(smiles)
    indexingVector=getMF_Index_Vector(MF)
    return indexingVector

def get_Index_Vector_with_Padding(MF,max_len):
    # Obtaining encoding vector according to compound morgan fingerprint
    Vector_with_Padding = [0] * max_len
    n_ones = 0
    for i in range(len(MF)):
        if MF[i] == 1:
            # The index starts from 1, because 0 is used to represent substructure-not-found
            Vector_with_Padding[n_ones] = i + 1
            n_ones += 1
    return np.array(Vector_with_Padding)

def txt_Str_data_to_IndexVec_npy(input_path):
    # Get the encoding vector of C-P pair from text sequence
    data = pd.read_csv(input_path, header=None)
    IndexVec_data=[]

    for index in range(len(data)):
        smiles, sequence, interaction = data.iloc[index, :]
        smile_MF=getMF(smiles)
        # drug encoding Vector
        smiles_IndexVec=get_Index_Vector_with_Padding(smile_MF,150)
        #protein encoding Vector
        sequence_IndexVec = label_sequence(sequence, 1000, CHARPROTSET)

        line = np.array([smiles_IndexVec, sequence_IndexVec, interaction])
        IndexVec_data.append(line)
    IndexVec_data=np.array(IndexVec_data)
    return IndexVec_data

def adjust_ratio_of_data(origin_data_path,ratio=1,output_path=''):
    # Adjust the proportion of positive and negative samples
    IndexVec_data=np.load(origin_data_path,allow_pickle=True)
    label_1_indexs=np.where(IndexVec_data[...,2] == 1)[0]
    label_0_indexs=np.where(IndexVec_data[...,2] == 0)[0]

    selectLabels_1 = label_1_indexs
    selectLabels_0 = random.sample(list(label_0_indexs), int(len(selectLabels_1)*ratio))
    selectLabels=np.r_[selectLabels_0,selectLabels_1]
    adjusted_data=IndexVec_data[selectLabels]

    np.save('1to{}_IndexVec_data.npy'.format(ratio),adjusted_data)
    return adjusted_data
