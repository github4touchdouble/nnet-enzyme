import sys
from numpy import array
import numpy as np
np.set_printoptions(threshold=sys.maxsize) # set numpy to print all values in array

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv() # load environment variables


def onehot(sequence):
    """
    method for one hot encoding a sequence, retuns concatenated np array 
    """
    seq_array = array(list(sequence)) 
    
    # integer encode input sequence: A list of integers, where each integer maps to a unique character in vocabulary [1,2,3,17,...]
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array) 
    
    onehot_encoder = OneHotEncoder(sparse_output=False)
    
    # here we take the integer encoded sequence and transform it to a one hot encoded sequence: [[0. 0. 1.], [...],...]
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    return np.concatenate(onehot_encoded_seq)


def create_ohe_seqs(df):
    """
    method that stdout the one hot encoded sequences 
    """
    # remove all unwanted AAs
    df = df[~df['Sequence'].str.contains('X')]
    df = df[~df['Sequence'].str.contains('O')]
    df = df[~df['Sequence'].str.contains('U')]
    
    for index, row  in df.iterrows():
        ohe_list = onehot(row['Sequence'])
        default_ohe_array = np.zeros((20440)) # 1022 * 20 = 20440 (max length of sequence * 20 possible AAs)
        for i in range(len(ohe_list)):
            if ohe_list[i] != 0.:
                default_ohe_array[i] += ohe_list[i] 
        print(f"{index}\t{default_ohe_array.tolist()}")
    

def filter_diff_multi_enzymes(df):
    """
    remove multi functional enzymes (enzymes with 2 ore more ec numbers that differ in their 1st 
    digit) from input dataframe
    """
    negative_df = df[df['EC number'].str.contains(';')]
    to_remove = []
    for ec in negative_df['EC number']:
        ec = ec.split(';')
        if ec[0][0] != ec[1][0]:
            to_remove.append(ec)
    df = df[~df['EC number'].isin(to_remove)]
    return df


def read_fasta_to_df(file) -> pd.DataFrame:
    """
    method for reading in fasta file and converting it to a pandas dataframe 
    """
    fasta_dict = {"Id": "Sequence"}
    with open(file, "r") as f:
        lines = f.readlines()
    current_key = ""
    for line in lines:
        line = line.strip("\n")
        if line[0] == ">":
            fasta_dict[line[1::]] = ""
            current_key = line[1::]
        else:
            fasta_dict[current_key] += line
    
    # Set the first row as column names
    df = pd.DataFrame.from_dict(fasta_dict, orient="index")
    df.columns = df.iloc[0]
    df = df[1:]
    return df


def collect_lengths(df) -> list[int]: 
    """
    This is used for extracting the length of the non enzme sequences, this is used for the cutoff at 1022 
    """
    lengths = []
    for seq in df['Sequence']:
        lengths.append(len(seq))
    return lengths


def main_enzymes(path_to_csv):
    """
    method used to convert enzymes to one hot encoded sequences
    """
    
    # read in csv
    enzymes = pd.read_csv(path_to_csv, sep=',')
    enzymes.set_index('Entry', inplace=True)

    # filter out multi functional enzymes
    filtered_enzymes = filter_diff_multi_enzymes(enzymes)

    # stdout one hot encoded sequences
    create_ohe_seqs(filtered_enzymes)


def main_non_enzymes(path_to_fasta):
    """
    method used to convert non_enzymes to one hot encoded sequences 
    """
    # read in fasta 
    non_enzyme = read_fasta_to_df(path_to_fasta)

    # remove all non_enzymes with sequence length > 1022
    non_enzyme['Sequence Length'] = collect_lengths(non_enzyme)
    non_enzyme = non_enzyme[non_enzyme['Sequence Length'] <= 1022] # Cutoff at 1022

    # stdout one hot encoded sequences
    create_ohe_seqs(non_enzyme)


if __name__ == "__main__":
    ENZYMES = os.getenv("ENZYMES")
    NON_ENZYMES = os.getenv("NON_ENZYMES")
    # main_enzymes(ENZYMES)
    # main_non_enzymes(NON_ENZYMES)
    print(onehot('MAGKQVRLVLLALGALVLLPTQGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL'))



