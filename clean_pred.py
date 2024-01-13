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
    class_labels_dict_l2 = {}
    class_labels_dict_l1 = {}


    for entry, pred in y_pred_raw:
        if ";" in pred:
            y_multi.append((entry, pred))
        else:
            digets = pred.split(".")
            level_2 = ".".join(digets[:2])
            level_2 = level_2.replace("EC:", "")
            level_1 = ".".join(digets[:1])
            level_1 = level_1.replace("EC:", "")
            class_labels_dict_l2[entry] = level_2
            class_labels_dict_l1[entry] = level_1

    # A dict which will be used to get the most confident label
    # for each entry id in the multiclass case
    # else it will just be the label

    for entry, pred in y_multi:
        ec_numbers = pred.split(";")
        set_ec_l2 = set()
        set_ec_l1 = set()
        confidences = []

        for ec in ec_numbers:
            ec = ec.strip()  # remove whitespace
            ec_conf = ec.split("/")  # split ec / confidence score
            ec = ec_conf[0]  # get ec
            ec = ec.replace("EC:", "")
            confidences.append(float(ec_conf[1]))  # get confidence score
            digets = ec.split(".")
            level_2 = ".".join(digets[:2])
            level_1 = ".".join(digets[:1])
            level_1 = level_1.strip()
            set_ec_l2.add(level_2)
            set_ec_l1.add(level_1)

        if len(set_ec_l2) > 1:
            index = np.argmax(confidences)
            ec = ec_numbers[index]
            digets = ec.split(".")
            level_2 = ".".join(digets[:2])
            level_2 = level_2.replace("EC:", "")
            class_labels_dict_l2[entry] = level_2
        if len(set_ec_l1) > 1:
            index = np.argmax(confidences)
            ec = ec_numbers[index]
            digets = ec.split(".")
            level_1 = ".".join(digets[:1])
            level_1 = level_1.replace("EC:", "")
            level_1 = level_1.strip()
            class_labels_dict_l1[entry] = level_1
        else:
            class_labels_dict_l2[entry] = list(set_ec_l2)[0]
            class_labels_dict_l1[entry] = list(set_ec_l1)[0]

    # print(class_labels_dict)
    # print(len(class_labels_dict))

    y_true_level_2 = []
    y_true_level_1 = []
    y_pred_level_2 = []
    y_pred_level_1 = []
    for entry, true in zip(trues['Entry'], trues['EC number']):
        digets = true.split(".")
        true_level_2_d = ".".join(digets[:2])
        true_level_2_d = true_level_2_d.replace("EC:", "")
        true_level_1_d = digets[0]
        y_true_level_2.append(true_level_2_d)
        y_pred_level_2.append(class_labels_dict_l2[entry])
        y_true_level_1.append(true_level_1_d)
        y_pred_level_1.append(class_labels_dict_l1[entry])

    print(len(y_true_level_1))
    print(len(y_pred_level_1))
    print(len(y_true_level_2))
    print(len(y_pred_level_2))

    print(class_labels_dict_l1)
    print(class_labels_dict_l2)

    print(y_true_level_1)

    y_true_level_2 = np.array(y_true_level_2)
    y_pred_level_2 = np.array(y_pred_level_2)
    y_true_level_1 = np.array(y_true_level_1)
    y_pred_level_1 = np.array(y_pred_level_1)

    scoring_funcs = [calculate_mcc_score, calculate_weighted_f1]

    np.savetxt("/home/malte/01_Documents/Temp/results_clean/y_true_l2_clean_new.txt", y_true_level_2, delimiter=',', fmt='%s')
    np.savetxt("/home/malte/01_Documents/Temp/results_clean/y_pred_l2_clean_new.txt", y_pred_level_2, delimiter=',', fmt='%s')
    np.savetxt("/home/malte/01_Documents/Temp/results_clean/y_true_l1_clean_new.txt", y_true_level_1, delimiter=',', fmt='%s')
    np.savetxt("/home/malte/01_Documents/Temp/results_clean/y_pred_l1_clean_new.txt", y_pred_level_1, delimiter=',', fmt='%s')


    # plot_bootstrapped_score([y_true_level_1, y_true_level_2], [y_pred_level_1, y_pred_level_2], scoring_funcs, ["CLEAN", "CLEAN"], 2)






if __name__ == "__main__":
    get_pred()

