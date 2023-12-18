import h5py
import pandas as pd
import numpy as np


def enzyme_split30_preprocessing(args):
    """
    :param args: DataFrame {Entry: <Entry>, EC number: <ecn>, Sequence: <seq>}
    :return: DataFrame {ID: <ID>, Enzyme class: <ec>, EC number: <ecn>, Sequence: <seq>}
    """

    def clean(args):
        # filter out enzymes with multiple ec numbers
        args = args[args["EC number"].str.contains(";") == False]
        args = args.reset_index(drop=True)

        # rename "Entry" to "ID"
        args.rename(columns={"Entry": "ID"}, inplace=True)
        return args

    def add_ec_number_column(args):
        # use first number of EC number to extract 1st class
        args["Enzyme class"] = args["EC number"].str.split('.').str[0].astype(int)

        return args

    args = clean(args)
    args = add_ec_number_column(args)

    return args


def read_h5(path_to_h5, prott5=True):
    """
    :param path_to_h5: FIle path as String
    :return: DataFrame {ID: <ID>. Embedding: [<emb>,<...>,...]}
    """

    bin = {"ID": [], "Embedding": []}
    with h5py.File(path_to_h5, 'r') as h:
        for id, emb in h.items():
            bin["ID"].append(id)
            if prott5:
                bin["Embedding"].append(list(emb)[0])
            else:
                bin["Embedding"].append(list(emb))

    return pd.DataFrame(bin)


def read_fasta(path_to_fasta):
    """
    :param path_to_fasta: FIle path as String
    :return: DataFrame {ID: <ID>. Sequence: <seq>}
    """

    bin = {"ID": [], "Sequence": []}
    with open(path_to_fasta, "r") as f:
        b = {"row_id": None, "row_seq": None}
        for row in f:
            if row.startswith(">") and b["row_id"] is None:
                b["row_id"] = row.rstrip()[1:]
            elif not row.startswith(">") and b["row_seq"] is None:
                b["row_seq"] = row.rstrip()
            else:
                raise ValueError(f"{b} and {row}")

            if b["row_id"] is not None and b["row_seq"] is not None:
                # copy/paste contents of b to bin
                bin["ID"].append(b["row_id"])
                bin["Sequence"].append(b["row_seq"])
                # reset b
                b["row_id"] = None
                b["row_seq"] = None

    return pd.DataFrame(bin)


def apply_embedding(args_embedding, args_proteins):
    """
    :param args_embedding: DataFrame {ID: <ID>, Embedding: [<emb>,<...>,...]}
    :param args_proteins: DataFrame {ID: <ID>, ...}
    :return: DataFrame {ID: <ID>, Embedding: [<emb>,<...>,...], ...}
    """

    bin = pd.merge(args_embedding, args_proteins, on="ID", how="inner")
    return bin
