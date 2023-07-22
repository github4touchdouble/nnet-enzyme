from numpy import array
import numpy as np
from numpy import argmax
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os
import sys
numpy.set_printoptions(threshold=sys.maxsize)

# method for one hot encoding a sequence, retuns concatenated np array
def onehot(sequence):
    seq_array = array(list(sequence)) 
    
    #integer encode the sequence
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array) 
    
    #one hot the sequence
    onehot_encoder = OneHotEncoder(sparse_output=False)
    
    #reshape because that's what OneHotEncoder likes
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    return np.concatenate(onehot_encoded_seq)

# don't use this, it's too slow, just left it here for reference
def create_ohe_df(df): 
    # remove all unwanted AAs
    df = df[~df['Sequence'].str.contains('X')]
    df = df[~df['Sequence'].str.contains('O')]
    df = df[~df['Sequence'].str.contains('U')]
    ohe_seqences = []
    
    
    for seq in df['Sequence']:
        ohe_list = onehot(seq)
        default_ohe_array = np.zeros((20440))
        for i in range(len(ohe_list)):
            default_ohe_array = np.insert(default_ohe_array, i, ohe_list[i])
        ohe_seqences.append(default_ohe_array)
    
    # create a dataframe from the one hot encoded sequences
    df['OneHotEncoded'] = ohe_seqences
    df['OneHotEncoded'] = df['OneHotEncoded'].apply(lambda x: x.tolist())
    # 20440
    return df


# method that stdout the one hot encoded sequences →  more efficient
def create_ohe_arr(df):
    # remove all unwanted AAs
    df = df[~df['Sequence'].str.contains('X')]
    df = df[~df['Sequence'].str.contains('O')]
    df = df[~df['Sequence'].str.contains('U')]
    
    
    for index, row  in df.iterrows():
        ohe_list = onehot(row['Sequence'])
        default_ohe_array = np.zeros((20440))
        for i in range(len(ohe_list)):
            if ohe_list[i] != 0.:
                default_ohe_array[i] += ohe_list[i] 
        print(f"{index}\t{default_ohe_array.tolist()}")
    

# remove multi functional enzymes
def filter_diff_multi_enzymes(df):
    negative_df = df[df['EC number'].str.contains(';')]
    to_remove = []
    for ec in negative_df['EC number']:
        ec = ec.split(';')
        if ec[0][0] != ec[1][0]:
            to_remove.append(ec)
    df = df[~df['EC number'].isin(to_remove)]
    return df


# method for reading in fasta file and converting it to a pandas dataframe
def read_fasta_to_df(file) -> pd.DataFrame:
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

# This is used for extracting the length of the non enzme sequences, this is used for the cutoff at 1022
def collect_lengths(df) -> list[int]: 
    lengths = []
    for seq in df['Sequence']:
        lengths.append(len(seq))
    return lengths

# NOTE: method I used to convert enzymes to one hot encoded sequences
def enzymes():
    path_to_csv = '~/Desktop/Dataset/data/enzymes/csv/split30.csv'
    output_path = '~/Desktop/Dataset/data/enzymes/multi_ez_filtered/ohe_spli30.csv'

    # read in csv
    enzymes = pd.read_csv(path_to_csv, sep=',')
    # 
    # filter out multi functional enzymes
    filtered_enzymes = filter_diff_multi_enzymes(enzymes)

    # create one hot encoded dataframe
    create_ohe_arr(filtered_enzymes)


# NOTE: method I used to convert non_enzymes to one hot encoded sequences
def non_enzymes():
    path_to_fasta = '/home/malte/Desktop/Dataset/data/non_enzyme/fasta/no_enzyme_train.fasta'
    output_path = '~/Desktop/Dataset/data/non_enzyme/ohe_train.csv'
    
    non_enzyme = read_fasta_to_df(path_to_fasta)
    non_enzyme['Sequence Length'] = collect_lengths(non_enzyme)
    non_enzyme = non_enzyme[non_enzyme['Sequence Length'] <= 1022] # Cutoff at 1022

    # non_enzymes_ohe = create_ohe_df(non_enzyme)
    # non_enzymes_ohe.to_csv(output_path, sep=',', index=False)
    create_ohe_arr(non_enzyme)


if __name__ == "__main__":
    enzymes()



