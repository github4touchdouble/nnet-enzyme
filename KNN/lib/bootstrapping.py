import math

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from scipy.stats import norm

def load_data():
    return load_digits(return_X_y=True)

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_predict(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def bootstrap_statistic(y_true, y_pred, statistic_func, B=10_000, alpha=0.05):
    bootstrap_scores = []
    for _ in range(B):
        indices = np.random.choice(len(y_pred), len(y_pred), replace=True)
        try:
            resampled_pred = y_pred[indices]
            resampled_true = y_true[indices]
            score = statistic_func(resampled_true, resampled_pred)
            bootstrap_scores.append(score)
        except:
            #print("Key error for " + str(indices))
            continue

    print(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)
    standard_error = np.std(bootstrap_scores, ddof=1)

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

if __name__ == "__main__":
    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=10_000, random_state=42),
        'Gaussian Naive Bayes': GaussianNB()
    }

    # Train classifiers, get predictions, and bootstrap F1 scores
    for name, clf in classifiers.items():
        y_pred = train_and_predict(clf, X_train, y_train, X_test)
        initial_f1 = calculate_f1(y_test, y_pred)
        mean_f1, se_f1, ci_95 = bootstrap_statistic(y_test, y_pred, calculate_f1)

        rounded_mean_f1, rounded_se_f1 = round_to_significance(mean_f1, se_f1)

        print(f"{name}:")
        # print(f"  - Initial F1 Score: {initial_f1:.2f}")
        print(f"  - Mean ± SE: {rounded_mean_f1} ± {rounded_se_f1}")
        # print(f"  - 95% CI: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
