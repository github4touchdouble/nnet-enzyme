import pandas as pd
import numpy as np

from metrics_helper_funcs.validation_tools import plot_bootstrapped_score, calculate_mcc_score, calculate_weighted_f1

PREDS_PATH = '/home/malte/Desktop/Dataset/Test_Data/new_clean.csv'
TRUES_PATH = '/home/malte/Desktop/Dataset/Test_Data/new.csv'


def get_pred():

    preds = pd.read_csv(PREDS_PATH, sep=',')
    trues = pd.read_csv(TRUES_PATH, sep=',')
    trues = trues[['Entry', 'EC number']]

    y_pred_raw = []

    for entry, pred in zip(preds['Identifier'], preds['Predicted EC Number']):
        y_pred_raw.append((entry, pred))

    y_multi = []
    class_labels_dict = {}

    for entry, pred in y_pred_raw:
        if ";" in pred:
            y_multi.append((entry, pred))
        else:
            digets = pred.split(".")
            level_2 = ".".join(digets[:2])
            level_2 = level_2.replace("EC:", "")
            class_labels_dict[entry] = level_2

    # A dict which will be used to get the most confident label
    # for each entry id in the multiclass case
    # else it will just be the label

    for entry, pred in y_multi:
        ec_numbers = pred.split(";")
        set_ec = set()
        confidences = []

        for ec in ec_numbers:
            ec = ec.strip()  # remove whitespace
            ec_conf = ec.split("/")  # split ec / confidence score
            ec = ec_conf[0]  # get ec
            ec = ec.replace("EC:", "")
            confidences.append(float(ec_conf[1]))  # get confidence score
            digets = ec.split(".")
            level_2 = ".".join(digets[:2])
            set_ec.add(level_2)

        if len(set_ec) > 1:
            index = np.argmax(confidences)
            ec = ec_numbers[index]
            digets = ec.split(".")
            level_2 = ".".join(digets[:2])
            level_2 = level_2.replace("EC:", "")
            class_labels_dict[entry] = level_2
        else:
            class_labels_dict[entry] = list(set_ec)[0]

    print(class_labels_dict)
    print(len(class_labels_dict))

    y_true_level_2 = []
    y_true_level_1 = []
    y_pred_level_2 = []
    y_pred_level_1 = []
    for entry, true in zip(trues['Entry'], trues['EC number']):
        digets = true.split(".")
        true_level_2 = ".".join(digets[:2])
        true_level_2 = true_level_2.replace("EC:", "")
        true_level_1 = digets[0]
        y_true_level_2.append(true_level_2)
        y_pred_level_2.append(class_labels_dict[entry])
        y_true_level_1.append(true_level_1)

    print(set(y_true_level_2))
    print(set(y_pred_level_1))

    y_true_level_2 = np.array(y_true_level_2)
    y_pred_level_2 = np.array(y_pred_level_2)

    y_true_level_1 = np.array(y_true_level_1)
    y_pred_level_1 = np.array(y_pred_level_1)

    scoring_funcs = [calculate_mcc_score, calculate_weighted_f1]

    np.savetxt("/home/malte/y_true_l2_clean.txt", y_true_level_2, delimiter=',', fmt='%s')
    np.savetxt("/home/malte/y_pred_l2_clean.txt", y_pred_level_2, delimiter=',', fmt='%s')


    np.savetxt("/home/malte/y_true_l1_clean.txt", y_true_level_2, delimiter=',', fmt='%s')
    np.savetxt("/home/malte/y_pred_l1_clean.txt", y_pred_level_2, delimiter=',', fmt='%s')


    # plot_bootstrapped_score([y_true_level_2], [y_pred_level_2], scoring_funcs, ["CLEAN"], 2)






if __name__ == "__main__":
    get_pred()

