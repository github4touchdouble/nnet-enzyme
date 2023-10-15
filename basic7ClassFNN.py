import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import h5py, sys, csv
from copy import deepcopy
import numpy as np

def compute_class_weights(labels):
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    class_weights = 1.0 / class_counts
    return class_weights

enzyme_sequences = []
listOfHeaders = []
ECHeaders = []

with h5py.File(r"D:/Uni/PBL/RNN/Dataset/split10_prott5.h5", "r") as hdf_handle:
    for header, emb in hdf_handle.items():
        np.set_printoptions(threshold=sys.maxsize)
        listOfHeaders.append(header)
        emb = np.array(list(emb))
        enzyme_sequences.append(emb[0])


csv_data = {}

# Read the CSV file and store its contents in the dictionary
with open('D:/Uni/PBL/RNN/Dataset/split10.csv', 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader, None)
    for row in csv_reader:
        header = row[0]
        ECNum = int(row[1][0])
        csv_data[header] = ECNum

# Now, you can look up the ECNum for each header in listOfHeaders efficiently
ECHeaders = [csv_data.get(header, None) for header in listOfHeaders]
#print(enzyme_sequences)
#print(listOfHeaders)
#print(ECHeaders)

# Convert data to PyTorch tensors
enzyme_sequences = np.array(enzyme_sequences)
enzyme_sequences = torch.Tensor(enzyme_sequences)
enzyme_labels = torch.LongTensor(ECHeaders)  # Assuming labels are integers
enzyme_labels -= 1                           # sub one to make pyTorch happy (OOB-Error)
#print(enzyme_sequences)
#print(enzyme_labels)


# Define your FNN model
class EnzymeClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(EnzymeClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes) - 1)
        ])
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.relu(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


# Define hyperparameters
input_size = 1024
num_classes = 7
learning_rate = 0.001
num_epochs = 100
hidden_sizes = [1024, 512, 256]  # Input layer size is 512, and there are 3 hidden layers with 256 neurons each.


# Initialize lists to store evaluation metrics for each fold
fold_accuracies = []
fold_f1_scores = []
fold_roc_aucs = []

# Initialize lists to store ROC curve data
all_fpr = []
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

# Define k-fold cross-validation
num_splits = 6
# You can adjust the number of folds
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

for fold, (train_indices, val_indices) in enumerate(skf.split(enzyme_sequences, enzyme_labels)):
    print(f"Fold {fold + 1}/{num_splits}")

    # Split the data into training and validation sets for this fold
    X_train, X_val = enzyme_sequences[train_indices], enzyme_sequences[val_indices]
    y_train, y_val = enzyme_labels[train_indices], enzyme_labels[val_indices]

    # Compute class weights for this fold's training data
    class_weights = compute_class_weights(y_train.numpy())

    # Create DataLoader for training and validation sets
    train_dataset = TensorDataset(torch.Tensor(X_train), y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Initialize the model
    model = EnzymeClassifier(input_size, hidden_sizes, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))

    # Initialize variables for early stopping
    best_val_loss = float("inf")
    patience = 5
    no_improvement_count = 0
    best_model_state = None  # Initialize best_model_state

    # Lists to store training and validation losses for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).tolist())
                val_labels.extend(labels.tolist())

        val_losses.append(val_loss / len(val_loader))
        val_accuracy = accuracy_score(val_labels, val_preds)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracy:.4f}")


        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            no_improvement_count = 0
            # Serialize the best model state using deepcopy
            best_model_state = deepcopy(model.state_dict())
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping: No improvement for {} epochs.".format(patience))
                break

        # Calculate ROC curve
    val_preds = []
    val_probs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_probs.extend(outputs.tolist())
            val_preds.extend(torch.argmax(outputs, dim=1).tolist())

    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)

    fpr, tpr, _ = roc_curve(val_labels, val_probs[:, 1], pos_label=1)
    all_fpr.append(fpr)
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC Fold {fold + 1}')

    # Load the best model for this fold
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate the model on the validation set for ROC curve and F1 score
    model.eval()
    val_preds = []
    val_probs = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            val_probs.extend(outputs.tolist())
            val_preds.extend(torch.argmax(outputs, dim=1).tolist())

    val_labels = np.array(val_labels)
    val_probs = np.array(val_probs)

    # Check validation labels for out-of-range values
    if any(label not in range(0, 7) for label in val_labels):
        raise ValueError("Validation labels contain out-of-range values.")

    fpr, tpr, _ = roc_curve(val_labels, val_probs[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    fold_roc_aucs.append(roc_auc)

    f1 = f1_score(val_labels, val_preds, average="weighted")
    fold_f1_scores.append(f1)

    accuracy = accuracy_score(val_labels, val_preds)
    fold_accuracies.append(accuracy)

    print(f"Fold {fold + 1} - ROC AUC: {roc_auc:.4f}, F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")

# Calculate and print mean ROC AUC, F1 Score, and Accuracy across all folds
mean_roc_auc = np.mean(fold_roc_aucs)
mean_f1_score = np.mean(fold_f1_scores)
mean_accuracy = np.mean(fold_accuracies)
print(f"Mean ROC AUC: {mean_roc_auc:.4f}, Mean F1 Score: {mean_f1_score:.4f}, Mean Accuracy: {mean_accuracy:.4f}")

# Calculate the mean ROC curve
mean_tpr /= num_splits
mean_auc = auc(mean_fpr, mean_tpr)

# Plot the mean ROC curve
plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})')

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Show the plot
plt.show()

# Plot training and validation losses for each fold
for i in range(num_splits):
    plt.plot(train_losses[i], label=f"Fold {i+1} Train Loss")
    plt.plot(val_losses[i], label=f"Fold {i+1} Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
