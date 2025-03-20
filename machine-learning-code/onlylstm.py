import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class YogaPoseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def load_multiple_files(base_dir, time_steps):
    all_features = []
    all_labels = []

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("yoga"):

            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):

                    label = folder_name
                    # Load data from the file
                    data = pd.read_csv(os.path.join(folder_path, file_name))
                    features = data.values

                    num_features = features.shape[1] // time_steps
                    features = features.reshape((features.shape[0], time_steps, num_features))
                    all_features.append(features)
                    all_labels.extend([label] * features.shape[0])

    all_features = np.vstack(all_features)
    return all_features, all_labels

file_directory = 'C:/Users/Malar/Desktop/Malarvizhi_Data/latest_data'
time_steps = 8

X, y = load_multiple_files(file_directory, time_steps)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
train_dataset = YogaPoseDataset(X_train, y_train)
test_dataset = YogaPoseDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
num_features = X_train.shape[2]

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, num_layers=2, num_classes=13):
        super(LSTMModel, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states for LSTM
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)  # 2 for num_layers
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        # Pass the output of the LSTM to the fully connected layer
        out = self.fc(out)
        return out


# Initialize and test the LSTM model with adjusted input
model = LSTMModel(input_size=num_features, hidden_size=64, num_layers=2, num_classes=13)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=60):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')


# Train the LSTM model
train_model(model, train_loader, criterion, optimizer)


# Evaluation loop
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


# Evaluate the LSTM model
evaluate_model(model, test_loader)

def plot_confusion_matrix(model, test_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Decode the labels back to original class names
    decoded_preds = label_encoder.inverse_transform(all_preds)
    decoded_labels = label_encoder.inverse_transform(all_labels)

    # Generate the confusion matrix
    cm = confusion_matrix(decoded_labels, decoded_preds, labels=label_encoder.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()


# Call the function to plot the confusion matrix
plot_confusion_matrix(model, test_loader, label_encoder)