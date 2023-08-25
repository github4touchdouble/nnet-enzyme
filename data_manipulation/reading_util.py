import pandas as pd
import h5py
import numpy as np


def read_fasta_to_df(file: str) -> pd.DataFrame:
    """
    method for reading in fasta file and converting it to a pandas dataframe
    :param file: Abs path to fasta file
    :return: A df with all seqs and ids
    """
    fasta_dict = {"Entry": "Sequence"}

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
    f.close()

    # Set the first row as column names
    df = pd.DataFrame.from_dict(fasta_dict, orient="index")
    df.columns = df.iloc[0]
    df = df[1:]

    return df


def filter_unwanted_seqs(df: pd.DataFrame, enzymes: bool) -> pd.DataFrame:
    """
    :param df: A dataframe containing either enzymes or non enzymes
    :param enzymes: If we pass a df containing enzymes we also need to filter out multifunctional enzymes
    :return: A filtered dataframe
    """

    # remove unwanted aas
    df = df[~df['Sequence'].str.contains('O')]
    df = df[~df['Sequence'].str.contains('U')]

    # for enzymes remove multifunctional enzymes
    if enzymes:
        multifunc_enzymes = df[df['EC number'].str.contains(';')]
        to_remove = []
        for ec in multifunc_enzymes['EC number']:
            ec = ec.split(';')
            if ec[0][0] != ec[1][0]:
                to_remove.append(ec)
        df = df[~df['EC number'].isin(to_remove)]
    else:

        df = df[df["Sequence"].apply(len) <= 1022] # if were working with non_enzymes we need to limit the sequence length

    return df


def read_esm2(path_to_esm2: str, is_enzyme: bool) -> pd.DataFrame:
    """
    :param path_to_esm2: Absolute path to esm2 file
    :return: A dataframe
    """
    with h5py.File(path_to_esm2) as hdf_handle:
        headers = []
        embeddings = []

        for header, emb in hdf_handle.items():
            headers.append(header)
            embeddings.append(np.array(list(emb)))

        df = pd.DataFrame(data={"Entry": headers, "Embedding": embeddings})

        if is_enzyme:
            df["Label"] = 1
        else:
            df["Label"] = -1

        return df


def load_ml_df(path_to_enzyme_esm2: str, path_to_non_enzyme_esm2: str) -> pd.DataFrame:
    """
    reads both esm2 embeddings (enzymes, non_enzymes) and concatenates to main machine learning df
    :param path_to_enzyme_esm2: Absolut path to enzyme esm2
    :param path_to_non_enzyme_esm2: Absolut path to non_enzyme esm2
    :return: A dataframe which can be the split to training and test data set
    """

    enzymes = read_esm2(path_to_enzyme_esm2, True)
    non_enzymes = read_esm2(path_to_non_enzyme_esm2, False)

    print(len(enzymes))
    print(len(non_enzymes))

    return pd.concat([enzymes, non_enzymes])
