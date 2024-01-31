import csv
import itertools

import h5py
import numpy as np
import sys
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.optim as optim
import keyboard
from itertools import chain
from bootstrapping import f1_score
from bootstrapping import calculate_f1, round_to_significance
from reading_util import filter_unwanted_esm2, H5Dataset
import time
import matplotlib.pyplot as plt

logger = SummaryWriter('runs/basic')
log_t = SummaryWriter('runs/training loss')
log_v = SummaryWriter('runs/validation loss')


def getDevice():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def computeClassWeights(labels):
    uniqueLabels, classCounts = np.unique(ar=labels, return_counts=True)
    class_weights = 1.0 / classCounts
    return class_weights


def loadData(path_to_esm2: str, path_to_enzyme_csv: str):
    """
    Reads in the embeddings and the EC numbers from the h5 file and the csv file and labels them accordingly.
    :param path_to_esm2: path to the h5 file
    :param path_to_enzyme_csv: path to the csv file
    :return: X: embeddings, y: EC numbers (labels)
    """

    to_remove = filter_unwanted_esm2(path_to_enzyme_csv, True)

    h5_dataset = H5Dataset(path_to_esm2, path_to_enzyme_csv)

    loader = torch.utils.data.DataLoader(h5_dataset, batch_size=32, shuffle=False)

    # Iterate over batches
    X = []
    y = []

    t0 = time.time()
    for batch in loader:
        emb, header, ec_numbers = batch
        if header not in to_remove:

            ec_class = [int(ec_number.split(".")[0]) for ec_number in ec_numbers]  # here we convert ec to int and do -1
            X.append(emb.numpy())
            y.extend(list(ec_class))

    # Convert the lists to numpy arrays
    enzymeSequences = np.vstack(X)
    ECHeaders = np.array(y)

    t1 = time.time()

    total = (t1 - t0) / 60

    print(f"LOG: Data loaded in: {round(total, 3)} min")
    print(f"LOG: ESM2 of enzymes: {len(enzymeSequences)}")
    print(f"LOG: Labels of enzymes: {len(ECHeaders)}")

    return enzymeSequences, ECHeaders


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv0 = nn.Conv1d(1, 4, 61, 1, 0)
        self.conv1 = nn.Conv2d(4, 16, (3, 3), 1, 0)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), 1, 1)
        self.conv3 = nn.Conv2d(32, 64, (3, 3), 1, 1)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 7)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x =         nn.functional.relu(self.conv0(x)) # 2560 -> 2500
        x = x.view(x.size(0), 4, 50, 50)       # 2560 -> 50x50
        x = self.mp(nn.functional.relu(self.conv1(x))) # 50x50 -> 48x48 -> 24x24
        x = self.mp(nn.functional.relu(self.conv2(x))) # 24x24 -> 24x24 -> 12x12
        x = self.mp(nn.functional.relu(self.conv3(x))) # 12x12 -> 12x12 -> 6x6
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sm(x)
        return x


# HYPERPARAMS
folds = 6
max_epoch = 100


class DataWrapper(Dataset):
    def __init__(self, sequences, lables):
        self.seq = sequences
        self.lab = lables

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx], self.lab[idx]


def predict(model, dataloader):
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.view(inputs.size(0), -1, 2560)
            outputs = model(inputs)
            all_outputs.extend(outputs)
    return all_outputs

def bootstrap_statistic(y_true, y_pred, statistic_func, B=10_000, alpha=0.05):
    bootstrap_scores = []
    for _ in range(B):
        indices = np.random.choice(len(y_pred), len(y_pred), replace=True)
        resampled_pred = y_pred[indices]
        resampled_true = y_true[indices]
        score = statistic_func(resampled_true, resampled_pred)
        bootstrap_scores.append(score)

    mean_score = np.mean(bootstrap_scores)
    standard_error = np.std(bootstrap_scores, ddof=1)

    # Calculate the 95% confidence interval
    lower_bound = np.percentile(bootstrap_scores, (alpha / 2) * 100)
    upper_bound = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return mean_score, standard_error, (lower_bound, upper_bound), bootstrap_scores


if __name__ == '__main__':
    a, b = loadData("D:/Uni/PBL/RNN/Dataset/split30_esm2_3b.h5", 'D:/Uni/PBL/RNN/Dataset/split30.csv')
    enzyme_sequences = np.array(a)
    enzyme_labels = np.array(b) - 1

    device = getDevice()
    print(f"Using {device} device")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(data=computeClassWeights(enzyme_labels), device=device, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters())
    print(model)

    timestep = 0
    for epoch in range(max_epoch):
        print(f"running epoch {epoch}")
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        for fold, (tInd, vInd) in enumerate(skf.split(enzyme_sequences, enzyme_labels)):
            trainDataset = DataWrapper(torch.as_tensor(data=enzyme_sequences[tInd], dtype=torch.float32, device=device),
                                       torch.as_tensor(data=enzyme_labels[tInd], dtype=torch.long, device=device))
            valDataset = DataWrapper(torch.as_tensor(data=enzyme_sequences[vInd], dtype=torch.float32, device=device),
                                     torch.as_tensor(data=enzyme_labels[vInd], dtype=torch.long, device=device))
            trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=0)
            valLoader = DataLoader(valDataset, batch_size=32, shuffle=False, num_workers=0)

            model.train()
            for miniBatchIndex, data in enumerate(trainLoader, 0):
                inputs, labels = data
                inputs = inputs.view(inputs.size(0), -1, 2560)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                timestep += 32
                log_t.add_scalar('losses', float(loss.item()), timestep)

            model.eval()
            with torch.no_grad():
                accLoss = 0.0
                accCount = 0
                correctByClass = torch.zeros(7, device=device)
                occByClass = torch.zeros(7, device=device)
                for _, v_data in enumerate(valLoader, 0):
                    v_inputs, v_labels = v_data
                    v_inputs = v_inputs.view(v_inputs.size(0), -1, 2560)
                    v_outputs = model(v_inputs)
                    v_loss = criterion(v_outputs, v_labels)
                    accLoss += v_loss.item()
                    accCount += 1

                    occByClass += torch.scatter(torch.zeros(32, 7, device=device), 1, v_labels.view(-1, 1), 1).sum(
                        dim=0)
                    pClass = torch.argmax(input=v_outputs, dim=1)
                    correct = torch.nonzero(torch.eq(pClass, v_labels))
                    correctByClass += torch.scatter(torch.zeros(len(correct), 7, device=device), 1, pClass[correct],
                                                    1).sum(dim=0)

                acc = correctByClass / occByClass
                logger.add_scalar('class-0-accuracy', acc[0], timestep)
                logger.add_scalar('class-1-accuracy', acc[1], timestep)
                logger.add_scalar('class-2-accuracy', acc[2], timestep)
                logger.add_scalar('class-3-accuracy', acc[3], timestep)
                logger.add_scalar('class-4-accuracy', acc[4], timestep)
                logger.add_scalar('class-5-accuracy', acc[5], timestep)
                logger.add_scalar('class-6-accuracy', acc[6], timestep)
                log_v.add_scalar('losses', accLoss / accCount, timestep)
            if keyboard.is_pressed('Esc'):
                break
        if keyboard.is_pressed('Esc'):
            break

    dataset = DataWrapper(torch.as_tensor(data=enzyme_sequences, dtype=torch.float32, device=device),
                          torch.as_tensor(data=enzyme_labels, dtype=torch.long, device=device))
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    scores = predict(model, loader)

    # Save the model's state dictionary to a file
    torch.save(model.state_dict(), 'D:/Uni/PBL/pbl_binary_classifier/KidaNN/CNN_esm.pth')

    predicted_classes = list(itertools.chain([torch.argmax(input=score, dim=0).cpu().tolist() for score in scores]))

    # Metrics: F1 Score and Accuracy
    f1 = f1_score(enzyme_labels, predicted_classes, average='weighted')
    accuracy = accuracy_score(enzyme_labels, predicted_classes)

    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    mean_score, standard_error, bounds, bootstrap_scores = bootstrap_statistic(np.array(enzyme_labels),
                                                                               np.array(predicted_classes),
                                                                               calculate_f1)

    # Set the range of values you want to plot (e.g., between 0.70 and 0.74)
    min_value = 0.8
    max_value = 0.99

    # Filter data within the specified range
    filtered_data = [x for x in bootstrap_scores if min_value <= x <= max_value]

    # Create the histogram
    plt.hist(filtered_data, bins=20, edgecolor='green')

    # Set the title and labels
    plt.title('Close-up Distribution Plot')
    plt.xlabel('Values')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()

    initial_f1 = calculate_f1(enzyme_labels, predicted_classes)
    rounded_mean_f1, rounded_se_f1 = round_to_significance(mean_score, standard_error)
    print(f"  - Initial F1 Score: {initial_f1:.2f}")
    print(f"  - Mean ± SE: {rounded_mean_f1} ± {rounded_se_f1}")
    print(f"  - 95% CI: [{bounds[0]:.2f}, {bounds[1]:.2f}]")

    print("Finished")

    print("Finished")
