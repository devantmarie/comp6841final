import pandas as pd
import torch
import torch.nn.functional as F
from Bio import SeqIO
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Classes to load the data. 
# ******************************
# here is the class to read the data. biopython to read and then transform?

def read_fasta_file(ffile):
    df = pd.DataFrame(columns = ["id","sequence","ylabel"])
    for record in SeqIO.parse(ffile,"fasta"):
         df.loc[len(df)] = [record.id,record.seq,record.description.split()[-1]]
    return(df)

#***************************************************

def list_y_classes_from_data(df):
    return df["ylabel"].unique() 

#***************************************************    
# map the rnalabel to number
def ncrna_map():
    nc_map= {'5S_rRNA':1, 
             '5_8S_rRNA':2, 
             'tRNA':3, 
             'ribozyme':4, 
             'CD-box':5, 
             'miRNA':6,  
             'Intron_gpI':7, 
             'Intron_gpII':8, 
             'HACA-box':9, 
             'riboswitch':10, 
             'IRES':11,
             'leader':12, 
             'scaRNA':13}
    
    nc_map_rev = {v: k for k, v in nc_map.items()}
    return nc_map,nc_map_rev
#***************************************************
# convert from sequecent to one hot.In case of 

def sequence_to_onehot(sequence):
    ncl_index =  {'A':0,'T':1,'G':2,'C':3}  
    sequence_indx = [ncl_index.get(i, -1) for i in sequence]
    onehot = []
    for j in sequence_indx:
        onehotvec = [0] * len(ncl_index)
        if j != -1:  
            onehotvec[j] = 1
        onehot.append(onehotvec)
    return torch.tensor(onehot)
    
#***************************************************

class NcRnaDataset(Dataset):
    def __init__(self,fastaFile,seq_length=120):
        self.nc_map, self.nc_map_rev = ncrna_map()
        self.ncdf = read_fasta_file(fastaFile)
        self.ncdf["y"] = self.ncdf["ylabel"].map(self.nc_map) 
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.ncdf)

    def __getitem__(self, idx):
        ncrna_onehot_a_seq = sequence_to_onehot(self.ncdf.iloc[idx]['sequence']) 
        label = self.ncdf.iloc[idx]['y']
        if self.seq_length !=0:
            ncrna_onehot_a_seq= self.resize_seqncr(ncrna_onehot_a_seq)
        return ncrna_onehot_a_seq, label 
    
    def resize_seqncr(self,nconehot):
        # Truncate or pad the sequence to the fixed length (max_length)
        seq_len = nconehot.shape[0]
        
        # Truncate if longer than max_length
        if seq_len > self.seq_length:
            nconehot = nconehot[:self.seq_length, :]  # Truncate to max_length
        
        # Pad if shorter than max_length
        elif seq_len < self.seq_length:
            padding = self.seq_length - seq_len
            nconehot = F.pad(nconehot, (0, 0, 0, padding), value=0)  

        return nconehot


#****************************************************


# The idea here was to dynamically adjust the size by batch, but the implementation becomes complex. 
# At least in the base class definition of `run`, it is not used. 
# Instead, the sequences are truncated or padded according to a sequence length ajustement (seq_lenght_ajust).

def collate_fn(batch):
    """
    Custom collate function to pad one-hot encoded DNA sequences.
    
    Args:
        batch (list of tuples or tensors): Each tuple may contain (sequence, label).
    
    Returns:
        torch.Tensor: Padded batch of one-hot encoded sequences.
        torch.Tensor: Sequence lengths before padding.
    """
    print(type(batch))
    iseq, ys = zip(*batch)
    maxlength = max([t.shape[0] for t in iseq])
    niseq= []
    for t in iseq:
        iseqcom = torch.zeros(maxlength-t.shape[0],4)
        t = torch.vstack((t, iseqcom))
        niseq.append(t)
   
    return list(zip(niseq,ys))
#******************************************************

