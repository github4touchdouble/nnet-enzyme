import pandas as pd
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import time


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


# def load_ml_df(path_to_enzyme_esm2: str, path_to_non_enzyme_esm2: str) -> pd.DataFrame:
#     """
#     reads both esm2 embeddings (enzymes, non_enzymes) and concatenates to main machine learning df
#     :param path_to_enzyme_esm2: Absolut path to enzyme esm2
#     :param path_to_non_enzyme_esm2: Absolut path to non_enzyme esm2
#     :return: A dataframe which can be the split to training and test data set
#     """
#
#     enzymes = read_esm2(path_to_enzyme_esm2, True)
#     non_enzymes = read_esm2(path_to_non_enzyme_esm2, False)
#
#     print(len(enzymes))
#     print(len(non_enzymes))
#
#     return pd.concat([enzymes, non_enzymes])



class H5Dataset(Dataset):
    def __init__(self, h5_file: str, csv_file: str):
        self.h5_file = h5_file
        self.ec_numbers = self._load_ec_numbers(csv_file)

        with h5py.File(self.h5_file, "r") as hdf_handle:
            self.length = len(hdf_handle.keys())
            self.keys = list(hdf_handle.keys())

    def _load_ec_numbers(self, csv_file: str) -> dict:
        df = pd.read_csv(csv_file)
        ec_dict = {}
        for _, row in df.iterrows():
            ec_dict[row["Entry"]] = row["EC number"]
        return ec_dict

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        with h5py.File(self.h5_file, "r") as hdf_handle:
            key = self.keys[index]
            emb = torch.tensor(hdf_handle[key], dtype=torch.float32).reshape(-1)
            header = key
            ec_number = self.ec_numbers.get(header, pd.NA)

        return emb, header, ec_number


class H5Dataset(Dataset):
    def __init__(self, h5_file: str, csv_file: str):
        self.h5_file = h5_file
        self.ec_numbers = self._load_ec_numbers(csv_file)

        with h5py.File(self.h5_file, "r") as hdf_handle:
            self.length = len(hdf_handle.keys())
            self.keys = list(hdf_handle.keys())

    def _load_ec_numbers(self, csv_file: str) -> dict:
        df = pd.read_csv(csv_file)
        ec_dict = {}
        for _, row in df.iterrows():
            ec_dict[row["Entry"]] = row["EC number"]
        return ec_dict

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        with h5py.File(self.h5_file, "r") as hdf_handle:
            key = self.keys[index]
            emb = torch.tensor(hdf_handle[key], dtype=torch.float32).reshape(-1)
            header = key
            ec_number = self.ec_numbers.get(header, pd.NA)

        return emb, header, ec_number


def load_ml_data_emb(path_to_esm2: str, path_to_enzyme_csv: str):
    """
    Reads in the embeddings and the EC numbers from the h5 file and the csv file and labels them accordingly.
    :param path_to_esm2: path to the h5 file
    :param path_to_enzyme_csv: path to the csv file
    :param class_depth: depth of the EC numbers
    :return: X: embeddings, y: EC numbers (labels)
    """

    h5_dataset = H5Dataset(path_to_esm2, path_to_enzyme_csv)

    loader = torch.utils.data.DataLoader(h5_dataset, batch_size=32, shuffle=True)

    # Iterate over batches
    X = []
    y = []

    t0 = time.time()

    for batch in loader:
        emb, _, ec_numbers = batch
        wanted_ec_class = [int(ec_number.split(".")[0]) - 1 for ec_number in ec_numbers] # here we convert ec to int and do -1
        
        X.append(emb.numpy())
        y.extend(list(wanted_ec_class))

    # Convert the lists to numpy arrays
    X = np.vstack(X)
    y = np.array(y)

    t1 = time.time()
    
    total = (t1-t0) / 60

    print(f"Data loaded in: {round(total, 3)} min")
    
    return X, y

# if __name__ == "__main__":
#     load_ml_data(path_to_esm2="/home/malte/Desktop/Dataset/data/enzymes/esm2/split10_esm2_3b.h5", path_tp_enzyme_csv="/home/malte/Desktop/Dataset/data/enzymes/csv/split10.csv", class_depth=2)

