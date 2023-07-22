from numpy import array
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


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
            default_ohe_array = np.insert(default_ohe_array, i, ohe_list.get(i))
        ohe_seqences.append(default_ohe_array)


    
    # create a dataframe from the one hot encoded sequences
    df['OneHotEncoded'] = ohe_seqences
    df['OneHotEncoded'] = df['OneHotEncoded'].apply(lambda x: x.tolist())
    # 20440
    return df

def filter_diff_multi_enzymes(df):
    """We only allow multi functional enzymes if they start with the same EC number
    →  3.4.23.25, 3.1.1.1 is okay but 3.2..., 2.1... is not

    :df: Dataframe : ID, EC number, Sequence
    :returns: A df without multi functional enzymes of different main classes

    """
    negative_df = df[df['EC number'].str.contains(';')]
    to_remove = []
    for ec in negative_df['EC number']:
        ec = ec.split(';')
        if ec[0][0] != ec[1][0]:
            to_remove.append(ec)
    df = df[~df['EC number'].isin(to_remove)]
    return df

def test():
    x = np.zeros(1000)
    y = [1.0 for i in range(1000) if i % 2 == 0]
    print(x)
    print(y)
    for i in range(len(y)):
        x = np.insert(x, i, y[i])

    print(x)


if __name__ == "__main__":
    path_to_csv = '~/Desktop/Dataset/data/enzymes/csv/split30.csv'
    output_path = '~/Desktop/Dataset/data/enzymes/multi_ez_filtered/ohe_spli30.csv'

    # read in csv
    df = pd.read_csv(path_to_csv, sep=',')
    # 
    # filter out multi functional enzymes
    filtered_df = filter_diff_multi_enzymes(df)

    # create one hot encoded dataframe
    filtered_ohe_df = create_ohe_df(filtered_df)


    # save to csv
    filtered_ohe_df.to_csv(output_path, sep=',', index=False)


