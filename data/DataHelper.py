import math
import json
import pickle
import numpy as np
import torch
from collections import OrderedDict

#pad: 0 -> Symbol that will fill in blank sequence if current batch data size is short than time steps
#vocab -> from DeepDTA

targetSeq_vocab = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
                   "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13,
                   "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19,
                   "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
targetSeq_vocabSize = 25

#vocab -> from DeepDTA
#Canonical SMILE
drugSeq_vocab = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
			      ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
			      "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
			      "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
			      "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
			      "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
			      "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
			      "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
			      "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
			      "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
			      "t": 61, "y": 62}
drugSeq_vocabSize = 62

'''
# Iso SMILE
CharIsoSmiSet = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33,
                 "/": 34, ".": 2, "1": 35, "0": 3, "3": 36, "2": 4,
                 "5": 37, "4": 5, "7": 38, "6": 6, "9": 39, "8": 7,
                 "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46,
                 "M": 47, "L": 13, "O": 48, "N": 14, "P": 15, "S": 49,
                 "R": 16, "U": 50, "T": 17, "W": 51, "V": 18, "Y": 52,
                 "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59,
                 "h": 24, "m": 60, "l": 25, "o": 61, "n": 26, "s": 62,
                 "r": 27, "u": 63, "t": 28, "y": 64}
CharIsoSmiLen = 64
'''
'''
#vocab -> from DeepAffinity
DrugDB_vocab = {"C": 1, "=": 2, "(": 3, ")": 4, "O": 5, "N": 6, "1": 7, "2": 8, "3": 9,
              "4": 10, "[": 11, "]": 12, "S": 13, "l": 14, "F": 15, "-": 16, "5": 17,
              "+": 18, ".": 19, "6": 20, "B": 21, "r": 22, "#": 23, "P": 24, "i": 25,
              "H": 26, "7": 27, "I": 28, "8": 29, "9": 30, "a": 31, "e": 32, "A": 33,
              "n": 34, "s": 35, "u": 36, "g": 37, "o": 38, "t": 39, "T": 40, "M": 41,
              "Z": 42, "b": 43, "K": 44, "R": 45, "d": 46, "W": 47, "G": 48, "L": 49,
              "c": 50, "h": 51, "V": 52, "m": 53, "E": 54, "Y": 55, "U": 56, "f": 57,
              "D": 58, "y": 59, "%": 60, "0": 61, "p": 62, "k": 63, "X": 64}
drugSeq_vocabSize = 64

targetSps_vocab = {"_PAD": 0, "CEDS": 1, "CEKS": 2, "CETS": 3, "BNGS": 4, "AEDM": 5, "CEDM": 6,
              "CEDL": 7, "AEKM": 8, "CEGS": 9, "CEKM": 10, "CETL": 11, "CETM": 12,
              "AEDL": 13, "AEKL": 14, "CEKL": 15, "ANGL": 16, "BNDS": 17, "BNTS": 18,
              "BNGM": 19, "ANGM": 20, "AETM": 21, "CEGM": 22, "AEDS": 23, "BNKS": 24,
              "CNGS": 25, "BEDS": 26, "AEGM": 27, "BNTM": 28, "AETL": 29, "CEGL": 30,
              "CNDS": 31, "ANTM": 32, "ANKM": 33, "ANDM": 34, "BNKM": 35, "CNTS": 36,
              "BEKS": 37, "BEKM": 38, "ANTL": 39, "BETS": 40, "AEKS": 41, "ANKL": 42,
              "BEDM": 43, "BNDM": 44, "CNGM": 45, "BETM": 46, "AEGL": 47, "CNKS": 48,
              "CNTM": 49, "BEGS": 50, "ANDL": 51, "ANGS": 52, "AETS": 53, "BEGM": 54,
              "ANDS": 55, "CNDM": 56, "AEGS": 57, "CNTL": 58, "CNGL": 59, "CNKM": 60,
              "ANTS": 61, "CNDL": 62, "ANKS": 63, "BNGL": 64, "CNKL": 65, "BEKL": 66,
              "BEDL": 67, "BETL": 68, "BNTL": 69, "BNKL": 70, "BNDL": 71, "BEGL": 72}
sps_vocabSize = 73
'''


#transfer token -> number
def seq_LabelDT(drug_seqs, target_seqs, drugSeq_maxlen, targetSeq_maxLen):
    drugSeq_truncated, targetSeq_truncated = drugSeq_maxlen, targetSeq_maxLen
    label_drugSeqs, label_targetSeqs = [], []

    if len(drug_seqs) >= drugSeq_truncated:
        for j in range(drugSeq_truncated):
            label_drug = drugSeq_vocab[drug_seqs[j].split()[0]]
            label_drugSeqs.append(label_drug)
    else:
        for j in range(len(drug_seqs)):
            label_drug = drugSeq_vocab[drug_seqs[j].split()[0]]
            label_drugSeqs.append(label_drug)

    if len(target_seqs) >= targetSeq_truncated:
        for j in range(targetSeq_truncated):
            label_traget = targetSeq_vocab[target_seqs[j].split()[0]]
            label_targetSeqs.append(label_traget)
    else:
        for j in range(len(target_seqs)):
            label_traget = targetSeq_vocab[target_seqs[j].split()[0]]
            label_targetSeqs.append(label_traget)

    return label_drugSeqs, label_targetSeqs



#transfer token -> number
def LabelDT(drug_seqs, target_seqs, drugSeq_maxlen, targetSeq_maxLen):
    label_drugSeqs, label_targetSeqs = [], []
    drugSeq_truncated, targetSeq_truncated = drugSeq_maxlen, targetSeq_maxLen

    for i in range(len(drug_seqs)):

        label_drugSeqs.append([])
        if len(drug_seqs[i]) >= drugSeq_truncated:
            for j in range(drugSeq_truncated):
                label_drug = drugSeq_vocab[drug_seqs[i][j].split()[0]]
                label_drugSeqs[i].append(label_drug)
        else:
            for j in range(len(drug_seqs[i])):
                label_drug = drugSeq_vocab[drug_seqs[i][j].split()[0]]
                label_drugSeqs[i].append(label_drug)

        label_targetSeqs.append([])
        if len(target_seqs[i]) >= targetSeq_truncated:
            for j in range(targetSeq_truncated):
                label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                label_targetSeqs[i].append(label_traget)
        else:
            for j in range(len(target_seqs[i])):
                label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                label_targetSeqs[i].append(label_traget)

        '''if sps==True:
            sps_buffer = target_seqs[i].split(',')

            if len(sps_buffer) >= targetSeq_truncated:
                for j in range(targetSeq_truncated):
                    label_traget = targetSps_vocab[sps_buffer[j]]
                    label_targetSeqs[i].append(label_traget)
            else:
                for j in range(len(sps_buffer)):
                    label_traget = targetSps_vocab[sps_buffer[j]]
                    label_targetSeqs[i].append(label_traget)
        else:
            if len(target_seqs[i]) >= targetSeq_truncated:
                for j in range(targetSeq_truncated):
                    label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                    label_targetSeqs[i].append(label_traget)
            else:
                for j in range(len(target_seqs[i])):
                    label_traget = targetSeq_vocab[target_seqs[i][j].split()[0]]
                    label_targetSeqs[i].append(label_traget)'''

    return label_drugSeqs, label_targetSeqs

#get compound + protein pairs
def GetPairs(label_drugSeqs, label_targetSeqs):

    pairs = []
    for i in range(len(label_targetSeqs)):
        drugSeq = label_drugSeqs[i]
        targetSeq = label_targetSeqs[i]
        pairs.append(drugSeq+targetSeq) # avoid ‘extend()’

    return pairs

#load davis and kiba
def LoadData(path, logspance_trans):

    print("Read %s start" % path)

    ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(path + "proteins.txt"), object_pairs_hook=OrderedDict)

    Y = pickle.load(open(path + "Y", "rb"), encoding='latin1')  # TODO: read from raw

    if logspance_trans:
        # Y = -(np.log10(Y / (math.pow(math.e, 9))))
        Y = -(np.log10(Y / (math.pow(10, 9))))

    XD = []
    XT = []

    for d in ligands.keys():
        XD.append(ligands[d])

    for t in proteins.keys():
        XT.append(proteins[t])

    return XD, XT, Y

#create samples for davis and kiba
def GetSamples(dataSet_name, drugSeqs, targetSeqs, affi_matrix):
    drugSeqs_buff, targetSeqs_buff, affiMatrix_buff= [], [], []
    if dataSet_name == 'davis':
        for i in range(len(drugSeqs)):
            for j in range(len(targetSeqs)):
                drugSeqs_buff.append(drugSeqs[i])
                targetSeqs_buff.append(targetSeqs[j])
                affiMatrix_buff.append(affi_matrix[i, j])

    if dataSet_name == 'kiba':
        for a in range(len(drugSeqs)):
            for b in range(len(targetSeqs)):
                if  ~(np.isnan(affi_matrix[a, b])):
                    drugSeqs_buff.append(drugSeqs[a])
                    targetSeqs_buff.append(targetSeqs[b])
                    affiMatrix_buff.append(affi_matrix[a, b])

    return drugSeqs_buff, targetSeqs_buff, affiMatrix_buff

#shuttle
def shuttle(drug, target, affini):

    drug = np.array(drug)
    target = np.array(target)
    affini = np.array(affini)
    index = [i for i in range(len(affini))]
    np.random.shuffle(index)

    shttle_drug = drug[index]
    shttle_target = target[index]
    shttle_affini = affini[index]

    return shttle_drug, shttle_target, shttle_affini


if __name__ == '__main__':
    path = '/home/robot/cyx/med/data_codes/data_davis/'
    XD, XT, Y = LoadData(path=path, logspance_trans=False)
    print(XD[:1], type(XD), len(XD))
    print(XT[:1], type(XT), len(XT))
    print(Y, type(Y), Y.shape)

    drugSeqs_buff, targetSeqs_buff, affiMatrix_buff = GetSamples('davis', drugSeqs=XD, targetSeqs=XT, affi_matrix=Y)
    print(len(drugSeqs_buff))
    print(len(targetSeqs_buff))
    print(len(affiMatrix_buff))
