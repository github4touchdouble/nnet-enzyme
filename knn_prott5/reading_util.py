import h5py
import pandas as pd


def enzyme_split30_preprocessing(args):
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


def read_h5(path_to_h5):
    """
    :param path_to_h5: FIle path as String
    :return: Dataframe {ID: <ID>. Embedding: [<emb>,<...>,...]}
    """

    bin = {"ID": [], "Embedding": []}
    with h5py.File(path_to_h5, 'r') as h:
        for id, emb in h.items():
            bin["ID"].append(id)
            bin["Embedding"].append(list(emb)[0])

    bin = pd.DataFrame(bin)

    return bin


def apply_prott5(args_prott5, args_enzymes):
    bin = {"ID": [], "Enzyme class": [], "EC number": [], "Embedding": [], "Sequence": []}
    for p5_row, p5_rec in args_prott5.iterrows():
        for enz_row, enz_rec in args_enzymes.iterrows():
            if enz_rec["ID"] == p5_rec["ID"]:
                bin["ID"].append(enz_rec["ID"])
                bin["Enzyme class"].append(enz_rec["Enzyme class"])
                bin["EC number"].append(enz_rec["EC number"])
                bin["Embedding"].append(p5_rec["Embedding"])
                bin["Sequence"].append(enz_rec["Sequence"])
    return bin


class Enzyme:
    def __init__(self, header, ec_class, ec_number, seq):
        self.header = header
        self.ec_class = ec_class
        self.ec_number = ec_number
        self.seq = seq


def read_enzyme_csv(path_to_csv: str) -> dict():
    enzymes_map = dict()
    with open(path_to_csv, "r") as path:
        line = path.readline()
        for line in path.readlines():
            field = line.strip().split(",")
            header = field[2]
            ec_class = field[1]
            ec_number = field[3]
            seq = field[4]
            enzymes_map[header] = Enzyme(header, ec_class, ec_number, seq)
    return enzymes_map
