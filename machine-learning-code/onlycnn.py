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

# Define the dataset class
class YogaPoseDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Load sensor data from multiple folders and CSV files
def load_multiple_files(base_dir, time_steps):
    all_features = []
    all_labels = []

    # Iterate over yoga1, yoga2, yoga3, etc., folders in the base directory
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("yoga"):
            # Iterate through CSV files inside each folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    # Extract label from the folder name (e.g., 'yoga1')
                    label = folder_name
                    # Load data from the file
                    data = pd.read_csv(os.path.join(folder_path, file_name))
                    features = data.values

                    # Reshape the data (samples, time_steps, num_features)
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

# Define CNN model only
class CNNOnlyModel(nn.Module):
    def __init__(self, num_features=8, num_classes=13, time_steps=4):
        super(CNNOnlyModel, self).__init__()

        # Adjust in_channels in conv1 to num_features to match the input shape
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * (time_steps // 2), 100)  # Adjust based on pool size
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # Permute to (batch size, num features, time steps)
        x = x.permute(0, 2, 1)

        # CNN with BatchNorm
        x = torch.relu(self.bn1(self.conv1(x)))

        x = torch.relu(self.conv2(x))

        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x




# Initialize and test the model with adjusted input
model = CNNOnlyModel(num_features=num_features, num_classes=13, time_steps=time_steps)
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