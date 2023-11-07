import math
import hhpy.plotting as hpt
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import matthews_corrcoef


# All methods used for validating the performance of our classifiers

# TODO confidence score; baseline
# TODO Acc; MCC multiclass;
# DONE: percent wise per row conf

def plot_confiusion_matrix(y_true, y_pred, plot_title, lable_to_class_dict=None, hide_inner_labels=False,
                           lable_size=10):
    if lable_to_class_dict == None:
        class_labels = set(y_true)
    else:
        class_labels = list(lable_to_class_dict.values())

    conf_matrix = confusion_matrix(y_true, y_pred)
    row_sums = conf_matrix.sum(axis=1)
    conf_matrix_percent = (conf_matrix.T / row_sums).T * 100

    # Create a confusion matrix heatmap
    plt.figure(figsize=(10, 7))

    # Plot the heatmap with percentwise coloring
    heatmap = sns.heatmap(conf_matrix_percent, annot=False, fmt=".2f", cmap="binary", cbar=False)

    # Loop through the confusion matrix and add text annotations
    if not hide_inner_labels:
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                text = f"{conf_matrix[i, j]}\n({conf_matrix_percent[i, j]:.2f}%)"
                text_color = 'white' if conf_matrix_percent[i, j] > 40 else 'black'
                plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', color=text_color, fontsize=14)


    plt.xlabel("Predicted", fontsize=lable_size)
    plt.ylabel("Actual", fontsize=lable_size)
    plt.xticks(fontsize=lable_size)
    plt.yticks(fontsize=lable_size)
    plt.yticks([i + 0.5 for i in range(len(class_labels))], class_labels, rotation=0)
    plt.title(f"{plot_title} (Percentwise Color)", fontsize=17)
    plt.show()


def plot_bootstrapped_score(y_trues, y_preds, scoring_funcs, model_names):
    score_df = pd.DataFrame(columns=["Model", "Metric", "Mean Score", "SE", "CI_0", "CI_1"])

    for i in range(len(y_trues)):

        data_to_append = []

        for func in scoring_funcs:
            y_true = y_trues[i]
            y_pred = y_preds[i]
            initial_metric, metric_name = func(y_true, y_pred)
            mean_metric, se_metric, ci_95 = bootstrap_statistic(y_true=y_true, y_pred=y_pred, statistic_func=func)
            rounded_mean_metric, rounded_se_metric = round_to_significance(mean_metric, se_metric)
            # Create a dictionary with the data
            data_row = {"Model": model_names[i], "Metric": metric_name, "Mean Score": rounded_mean_metric,
                        "SE": rounded_se_metric, "CI_0": ci_95[0],
                        "CI_1": ci_95[1]}
            data_to_append.append(data_row)

        score_df = pd.concat([score_df, pd.DataFrame(data_to_append)])

    # Set the style and context for the plot
    sns.set()
    sns.set(font_scale=1.5)  # Adjust font size as needed
    sns.set(style="whitegrid")
    # color_palette = ["#2aa5a5", "#fc4b00", "#7647fa", "#ffe512", "#ed174f", "#008365", "#c2837a"]
    color_palette = ["#b2182b", "#ef8a62", "#fddbc7", "#f7f7f7", "#d1e5f0", "#67a9cf", "#2166a"]
    color_palette = [
        "#d73027",  "#91bfdb", "#fc8d59", "#4575b4", "#d8b365", "#5ab4ac", "#af8dc3"
    ]

    # Create the bar plot with custom error bars and hue="Model"
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Metric", y="Mean Score", hue="Model", data=score_df, **{'width': 0.3}, palette=color_palette)

    # Customize the plot labels
    ax.set(xlabel="Metric", ylabel="Mean Score")
    ax.set_title("Performance Metrics with Custom Error Bars by Model")

    plt.legend(title="Model", loc="upper left", bbox_to_anchor=(1, 1))
    print(score_df)

    # Manually add error bars using the calculated bounds
    for i, bar in enumerate(ax.patches):
        metric_data = score_df.iloc[i]
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        lower_bound = metric_data["CI_0"]
        upper_bound = metric_data["CI_1"]

        plt.plot([x, x], [lower_bound, upper_bound], color="black")
        plt.plot([x - 0.01, x + 0.01], [lower_bound, lower_bound], color="black")
        plt.plot([x - 0.01, x + 0.01], [upper_bound, upper_bound], color="black")

    plt.tight_layout()
    plt.show()


####################### Helper funcs, don't use them, the main funcs use them ########################################

def bootstrap_statistic(y_true, y_pred, statistic_func, B=10_000, alpha=0.05):
    bootstrap_scores = []
    for _ in range(B):
        indices = np.random.choice(len(y_pred), len(y_pred), replace=True)
        resampled_pred = y_pred[indices]
        resampled_true = y_true[indices]
        score, _ = statistic_func(resampled_true, resampled_pred)
        bootstrap_scores.append(score)

    mean_score = np.mean(bootstrap_scores)
    standard_error = np.std(bootstrap_scores, ddof=1)

    # Set the range of values you want to plot (e.g., between 0.70 and 0.74)
    min_value = min(bootstrap_scores)
    max_value = max(bootstrap_scores)

    # Filter data within the specified range
    filtered_data = [x for x in bootstrap_scores if min_value <= x <= max_value]

    # Calculate the 95% confidence interval
    lower_bound = np.percentile(bootstrap_scores, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return mean_score, standard_error, (lower_bound, upper_bound)


def round_to_significance(x, significance):
    if significance == 0.0:
        sig_position = 0
    else:
        sig_position = int(math.floor(math.log10(abs(significance))))
    return round(x, -sig_position), round(significance, -sig_position + 1)


def calculate_micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro'), "Micro F1-Score"


def calculate_macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro'), "Macro F1-Score"

def calculate_weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted'), "Weighted F1-Score"


def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred), "Accuracy"


if __name__ == '__main__':
    y_true_model_1 = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    y_pred_model_1 = np.array(
        [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1])

    y_true_model_2 = np.array(
        [0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2,
         2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])
    y_pred_model_2 = np.array(
        [1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 2, 1, 0, 0, 2,
         0, 2, 0, 1, 0, 1, 1, 2, 2, 1, 1, 0, 1, 2, 0, 2, 1, 2, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 3, 5, 5, 3, 5, 1, 0, 4, 6,
         6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    y_true_model_3 = np.array(
        [0, 1, 3, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2,
         2, 3, 1, 1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 1, 3, 2, 1, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
         6, 6, 6, 6, 5, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    y_pred_model_3 = np.array(
        [0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 2, 1, 0, 0, 2,
         0, 2, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 2, 1, 2, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 3, 5, 5, 3, 5, 1, 0, 4, 6,
         6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 4, 4, 3, 5, 6, 6, 6, 6, 6])

    y_true_model_4 = np.array(
        [0, 1, 3, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2,
         2, 3, 1, 1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 1, 3, 2, 1, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
         6, 6, 6, 6, 5, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    y_pred_model_4 = np.array(
        [0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 2, 1, 0, 0, 2,
         1, 3, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 3, 2, 3, 1, 5, 6, 6, 4, 5, 6, 4, 5, 6, 3, 5, 5, 3, 5, 1, 0, 4, 6,
         6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 5, 5, 3, 5, 6, 6, 6, 6, 6])

    y_true_model_5 = np.array(
        [0, 1, 3, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2,
         2, 3, 1, 1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 1, 3, 2, 1, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
         6, 6, 6, 6, 5, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    y_pred_model_5 = np.array(
        [0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 2, 1, 0, 0, 2,
         1, 3, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 3, 2, 3, 1, 5, 6, 6, 4, 5, 6, 4, 5, 6, 3, 5, 5, 3, 5, 1, 0, 4, 6,
         6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 5, 5, 3, 5, 6, 6, 6, 6, 6])
    y_true_model_6 = np.array(
        [0, 1, 3, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2,
         2, 3, 1, 1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 1, 3, 2, 1, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
         6, 6, 6, 6, 5, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    y_pred_model_6 = np.array(
        [0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 2, 1, 0, 0, 2,
         1, 3, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 3, 2, 3, 1, 5, 6, 6, 4, 5, 6, 4, 5, 6, 3, 5, 5, 3, 5, 1, 0, 4, 6,
         6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 5, 5, 3, 5, 6, 6, 6, 6, 6])


    y_true_model_7 = np.array(
        [0, 1, 3, 2, 1, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 3, 1, 1, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2,
         2, 3, 1, 1, 1, 1, 2, 4, 3, 2, 2, 2, 2, 1, 3, 2, 1, 1, 0, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
         6, 6, 6, 6, 5, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6])

    y_pred_model_7 = np.array(
        [0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 3, 2, 1, 1, 1, 0, 2, 0, 3, 0, 1, 0, 1, 1, 3, 2, 2, 1, 0, 0, 2,
         1, 3, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 1, 3, 2, 3, 1, 5, 6, 6, 4, 5, 6, 4, 5, 6, 3, 5, 5, 3, 5, 1, 0, 4, 6,
         6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 4, 4, 5, 5, 3, 5, 6, 6, 6, 6, 6])

    lable_to_class = {
        0: "Non Enzyme",
        1: "Enzyme"
    }

    lable_to_multiclass = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
        5: 6,
        6: 7
    }

    y_trues = [y_true_model_1, y_true_model_2, y_true_model_3, y_true_model_4, y_true_model_5, y_true_model_6, y_true_model_7]
    y_preds = [y_pred_model_1, y_pred_model_2, y_pred_model_3, y_pred_model_4, y_pred_model_5, y_pred_model_6, y_pred_model_7]

    metric_funcs = [calculate_accuracy, calculate_micro_f1, calculate_macro_f1]
    model_names = ["Binary", "FNN 1", "FNN 2", "O.o", "f", "g", "l"]  # Names used for plotting

    plot_bootstrapped_score(y_trues, y_preds, scoring_funcs=metric_funcs, model_names=model_names)
