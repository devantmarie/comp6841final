import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random 
from Bio import SeqIO
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

"""
Classes to load and preprocess biological sequence data.
********************************************************

Utilities and dataset class for loading and preprocessing non-coding RNA (ncRNA) sequences from FASTA files. Includes functions for reading data (requires Biopython), mapping labels, converting sequences to one-hot encoding, applying reverse complement transformations, and visualizing class and length distributions. The `NcRnaDataset` class prepares data for training by resizing sequences and optionally combining multiple FASTA sources.
"""
#**************************************************

def read_fasta_file(ffile):
    """
    Reads a FASTA file using Biopython and converts it into a pandas DataFrame (label extracted from the sequence description).
    """
    df = pd.DataFrame(columns = ["id","sequence","ylabel"])
    for record in SeqIO.parse(ffile,"fasta"):
         df.loc[len(df)] = [record.id,record.seq,record.description.split()[-1]]
    return(df)

#***************************************************

def list_y_classes_from_data(df):
    return df["ylabel"].unique() 

#***************************************************    

def ncrna_map():
    """
    Returns two dictionaries for ncRNA class mapping:
    - nc_map: maps ncRNA class names to numeric labels.
    - nc_map_rev: reverse mapping from numeric labels back to class names.
    """
    nc_map= {'unknown':0,
             '5S_rRNA':1, 
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
#
def sequence_to_onehot(sequence):
    """
    Converts an RNA sequence into a one-hot encoded tensor.
        Each nucleotide (A, T, G, C) is encoded into a 4-element binary vector.
        Invalid or ambiguous bases (e.g., N, Y, R) are encoded as [0, 0, 0, 0].
    Returns:
        torch.Tensor: One-hot encoded representation of the input sequence.
    """
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
#
def hist_plot_ncrna_class(df):
    """
    Displays a histogram of ncRNA class distribution from the given DataFrame.
    Parameters:
        df (pandas.DataFrame): DataFrame containing a 'ylabel' column with ncRNA class labels.
    Returns:
        matplotlib.pyplot: The plot object for further customization if needed.
    """
    class_counts = df['ylabel'].value_counts()
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("ncRNA Class")
    plt.ylabel("Count")
    plt.title("Histogram of Non-Coding RNA by Class")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    return plt

#***************************************************
# 
def dist_seq_lenghts(df):
    """
    Displays a histogram showing the distribution of ncRNA sequence lengths.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing a 'real_sequence_length' column.
    
    Returns:
        matplotlib.pyplot: The plot object for further customization if needed.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['real_sequence_length'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title("Distribution of ncRNA Sequence Lengths")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    return plt

#***************************************************
# 
def reverse_complement(rna_sequence):
    """
    Compute the reverse complement of an RNA/DNA sequence.
    This function handles both standard and ambiguous nucleotide codes.
    It maps each base to its complement and reverses the sequence.
    Parameters:
        rna_sequence (str): The nucleotide sequence (RNA or DNA) as a string.
    Returns:
        str: The reverse complement of the input sequence.
    """
    complement_base= {
        'A': 'T',
        'T': 'A',
        'C': 'G',
        'G': 'C',
        'N': 'N',  # Undetermined base 
        'R': 'Y',  # Purine (A or G)
        'Y': 'R',  # Pyrimidine (C or T)
        'S': 'S',  # (G or C)
        'W': 'W',  # (A or T)
        'K': 'M',  # (G or T)
        'M': 'K',  # (A or C)
        'B': 'V',  # (C, G, or T)
        'D': 'H',  # (A, G, or T)
        'H': 'D',  # (A, C, or T)
        'V': 'B',  # (A, C, or G)
        'X': 'X'   # Unknown base
                
    }
    complement = ''.join([complement_base[base] for base in rna_sequence])
    reverse_complement = complement[::-1]
    return reverse_complement

#*************************************************
# 
class NcRnaDataset(Dataset):
    """
    Custom Dataset class for loading, processing, and transforming non-coding RNA (ncRNA) sequences from FASTA files. 
    It extends the PyTorch `Dataset` class to be used in a `DataLoader`. The class supports reverse complement transformations, 
    one-hot encoding of sequences, and resizing sequences to a fixed length. It also includes methods for plotting histograms 
    of class distributions and sequence lengths.
    """
        
    def __init__(self,fastaFile,seq_length=120,random_rev_compl_transform_prob = 0,fastaFile2=None, fastaFile3=None):
        """
        Initializes the NcRnaDataset class by loading ncRNA sequences from one or more FASTA files, mapping labels to numerical values, 
        and computing sequence lengths. It also supports sequence resizing and random reverse complement transformations.
        
        Parameters:
        - fastaFile (str): Path to the primary FASTA file containing the ncRNA sequences.
        - seq_length (int, optional): The fixed length to which sequences will be resized. Default is 120.
        - random_rev_compl_transform_prob (float, optional): The probability of applying a random reverse complement transformation to the sequences. Default is 0.
        - fastaFile2 (str, optional): Path to an additional FASTA file to be concatenated with the first one. Default is None.
        - fastaFile3 (str, optional): Path to a third FASTA file to be concatenated with the first one. Default is None.
        """
        self.nc_map, self.nc_map_rev = ncrna_map()
        self.ncdf = read_fasta_file(fastaFile)
        if fastaFile2 is not None:
            ncdf2 = read_fasta_file(fastaFile2)
            self.ncdf = pd.concat([self.ncdf,ncdf2],ignore_index=True)
        if fastaFile3 is not None:
            ncdf3 = read_fasta_file(fastaFile3)
            self.ncdf = pd.concat([self.ncdf,ncdf3],ignore_index=True)
        self.ncdf["y"] = self.ncdf["ylabel"].map(self.nc_map) 
        self.ncdf['real_sequence_length'] = self.ncdf['sequence'].apply(len)
        self.seq_length = seq_length
        self.random_rev_compl_transform_prob = random_rev_compl_transform_prob

     #*********************************************
        
    def __len__(self):
        return len(self.ncdf)

    #*********************************************

    def __getitem__(self, idx):
        ncrna_seq = self.ncdf.iloc[idx]['sequence']
        if random.random() <  self.random_rev_compl_transform_prob:
             ncrna_seq = reverse_complement(ncrna_seq)
        ncrna_onehot_a_seq = sequence_to_onehot(ncrna_seq) 
        label = self.ncdf.iloc[idx]['y']
        if self.seq_length !=0:
            ncrna_onehot_a_seq= self.resize_seqncr(ncrna_onehot_a_seq)
        return ncrna_onehot_a_seq, label 

    #*********************************************
    
    def resize_seqncr(self,nconehot):
        seq_len = nconehot.shape[0]
        if seq_len > self.seq_length:
            nconehot = nconehot[:self.seq_length, :]  
        elif seq_len < self.seq_length:
            padding = self.seq_length - seq_len
            nconehot = F.pad(nconehot, (0, 0, 0, padding), value=0)  
        return nconehot

    #*********************************************
    
    def hist_nc_class(self):
        return hist_plot_ncrna_class(self.ncdf)

    #*********************************************
    
    def dist_nc_len(self):
        return dist_seq_lenghts(self.ncdf)

    #*********************************************


