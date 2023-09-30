import pandas as pd
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import time


def read_esm2(path_to_esm2: str, is_enzyme: bool, ignore_headers: list) -> pd.DataFrame:
    """
    :param path_to_esm2: Absolute path to esm2 file
    :return: A dataframe
    """
    with h5py.File(path_to_esm2) as hdf_handle:
        headers = []
        embeddings = []

        for header, emb in hdf_handle.items():
            if header not in ignore_headers:
                headers.append(header)
                embeddings.append(np.array(list(emb)))

        df = pd.DataFrame(data={"Entry": headers, "Embedding": embeddings})

        if is_enzyme:
            df["Label"] = 1
        else:
            df["Label"] = -1

        return df


def read_fasta_to_df(file: str) -> pd.DataFrame:
    """
    method for reading in fasta file and converting it to a pandas dataframe
    :param file: Abs path to fasta file
    :return: A df with all seqs and ids
    """
    fasta_dict = {}

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
    df = pd.DataFrame(fasta_dict.items(), columns=['Entry', 'Sequence'])

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
        for multi_ec in multifunc_enzymes['EC number']:
            ec = multi_ec.split(';')
            test_set = set()

            for i in range(len(ec)):
               test_set.add(ec[i][0])

            if len(test_set) > 1: # If this len(set) != 1 we have a multi func enzyme of different main classes
                to_remove.append(ec)

        df = df[~df['EC number'].isin(to_remove)]

    else:
        df = df[
            df["Sequence"].apply(len) <= 1022]  # if were working with non_enzymes we need to limit the sequence length

    return df


def filter_unwanted_esm2(path_to_csv: str, is_enzyme: bool):
    """
    :param path_to_csv: Absolute path to csv file
    :param is_enzyme: If we pass enzymes we also need to filter out multifunctional enzymes
    :return: A list of headers to ignore when reading in the esm2 embeddings
    """
    if is_enzyme:
        df = pd.read_csv(path_to_csv, sep=",")
    else:
        df = read_fasta_to_df(path_to_csv)
    to_remove = []

    remove_O = df[df['Sequence'].str.contains('O')]["Entry"]
    remove_U = df[df['Sequence'].str.contains('U')]["Entry"]

    print(f"LOG: {len(remove_O)} Sequences with aa O in {path_to_csv}")
    print(f"LOG: {len(remove_U)} Sequences with aa U in {path_to_csv}")

    to_remove.extend(remove_O.to_list())
    to_remove.extend(remove_U.to_list())

    # for enzymes remove multifunctional enzymes
    if is_enzyme:
        multifunc_enzymes = df[df['EC number'].str.contains(';')]
        remove_multif = []
        for index, row in multifunc_enzymes.iterrows():

            ec = row["EC number"].split(';') # split the ec numbers by ; meaning ec = ["1.2.3.4", "2.2.3.44"] for 1.2.3.4;2.2.3.44. This would be removed
            header = row["Entry"]
           
            test_set = set()
            for i in range(len(ec)):
               test_set.add(int(ec[i][0]))

            if len(test_set) > 1: # If this len(set) != 1 we have a multi func enzyme of different main classes
                remove_multif.append(header)

        to_remove.extend(remove_multif)
        print(f"LOG: {len(remove_multif)} multifunctional enzymes with diff ec main classes in {path_to_csv}")

    else:
        remove_seq_cutoff = df[df["Sequence"].apply(len) > 1022]["Entry"]  # if were working with non_enzymes we need to limit the sequence length
        print(f"LOG: {len(remove_seq_cutoff)} non enzymes are longer than 1022 cutoff")
        to_remove.extend(remove_seq_cutoff.to_list())

    print(f"LOG: {len(to_remove)} entries will be ignored")
    return to_remove


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


class H5ESM2(Dataset):
    def __init__(self, h5_file: str):
        self.h5_file = h5_file

        with h5py.File(self.h5_file, "r") as hdf_handle:
            self.length = len(hdf_handle.keys())
            self.keys = list(hdf_handle.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        with h5py.File(self.h5_file, "r") as hdf_handle:
            key = self.keys[index]
            emb = torch.tensor(hdf_handle[key], dtype=torch.float32).reshape(-1)
            header = key

        return emb, header


def load_ml_data_emb(path_to_esm2: str, path_to_enzyme_csv: str):
    """
    Reads in the embeddings and the EC numbers from the h5 file and the csv file and labels them accordingly.
    :param path_to_esm2: path to the h5 file
    :param path_to_enzyme_csv: path to the csv file
    :return: X: embeddings, y: EC numbers (labels)
    """

    to_remove = filter_unwanted_esm2(path_to_enzyme_csv, True)

    h5_dataset = H5Dataset(path_to_esm2, path_to_enzyme_csv)

    loader = torch.utils.data.DataLoader(h5_dataset, batch_size=32, shuffle=True)

    # Iterate over batches
    X = []
    y = []

    t0 = time.time()

    for batch in loader:
        emb, header, ec_numbers = batch
        if header not in to_remove:
            ec_class = [int(ec_number.split(".")[0]) - 1 for ec_number in ec_numbers]  # here we convert ec to int and do -1

            X.append(emb.numpy())
            y.extend(list(ec_class))

    # Convert the lists to numpy arrays
    X = np.vstack(X)
    y = np.array(y)

    t1 = time.time()

    total = (t1 - t0) / 60

    print(f"LOG: Data loaded in: {round(total, 3)} min")
    print(f"LOG: ESM2 of enzymes: {len(X)}")
    print(f"LOG: Labels of enzymes: {len(X)}")

    return X, y


def load_non_enz_esm2(non_enzymes_fasta_path: str, non_enzymes_esm2_path: str):
    """
    Used for reading in esm2 embeddings of non_enzymes, since we don't have a .csv file
    Filters sequences longer than cutoff at 1022, sequences containing either O od U as aa
    :param non_enzymes_fasta_path: Path to non_enzyme_fasta
    :param non_enzymes_esm2_path: Path to non_enzyme_esm2
    :return: X_neg, y_neg
    """

    non_enz_to_remove = filter_unwanted_esm2(non_enzymes_fasta_path, False)

    h5_esm2 = H5ESM2(non_enzymes_esm2_path)

    loader = torch.utils.data.DataLoader(h5_esm2, batch_size=32, shuffle=True)

    # Iterate over batches
    X_neg = []

    t0 = time.time()

    for batch in loader:
        emb, header = batch
        if header not in non_enz_to_remove:
            X_neg.append(emb.numpy())

    t1 = time.time()

    # Convert the lists to numpy arrays
    X_neg = np.vstack(X_neg)

    y_neg = [7 for _ in range(len(X_neg))]  # adding labels for non enzymes:
    # (0-6 → enzyme; 7 → non_enzyme)
    y_neg = np.array(y_neg)

    total = (t1 - t0) / 60

    print(f"LOG: Non Enzymes data loaded in: {round(total, 3)} min")
    print(f"LOG: ESM2 of non enzymes: {len(X_neg)}")
    print(f"LOG: Labels of non enzymes: {len(y_neg)}")

    return X_neg, y_neg
