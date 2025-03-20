import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
                    num_features = features.shape[1] // time_steps  # Ensure num_features is calculated correctly
                    print(f"Processing {file_name} with {num_features} features per {time_steps} time steps.")

                    if num_features == 0:
                        raise ValueError(f"num_features cannot be 0. Check the columns in the data "
                                         f"({features.shape[1]}) vs. time_steps ({time_steps}).")

                    features = features.reshape((features.shape[0], time_steps, num_features))

                    # Append the features and corresponding labels
                    all_features.append(features)
                    all_labels.extend([label] * features.shape[0])  # Assign the same label to all samples in the file

    # Concatenate all features and labels into a single dataset
    all_features = np.vstack(all_features)
    return all_features, all_labels


# Path to the folder containing your yoga pose CSV files
file_directory = 'C:/Users/Malar/Desktop/Malarvizhi_Data/latest_data'

time_steps = 8  # Define your time steps to fit 32 columns

# Load the data from multiple CSV files
X, y = load_multiple_files(file_directory, time_steps)

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize the scaler
scaler = MinMaxScaler()

# Fit the scaler on the features of the entire dataset (X) before splitting
X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Now split the scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Create PyTorch Dataset and DataLoader
train_dataset = YogaPoseDataset(X_train, y_train)
test_dataset = YogaPoseDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Now calculate num_features from the reshaped data
num_features = X_train.shape[2]


# Define the CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, num_features, num_classes=13):
        super(CNNLSTM, self).__init__()
        # CNN layers with BatchNorm
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)  # Batch Normalization after first conv layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True,
                            dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # 1. CNN Layers with Batch Normalization
        x = x.permute(0, 2, 1)  # Rearrange dimensions to (batch_size, num_features, time_steps)
        x = torch.relu(self.bn1(self.conv1(x)))  # Convolution + BatchNorm + ReLU activation on Conv1 layer
        x = torch.relu(self.conv2(x))  # Convolution + ReLU activation on Conv2 layer
        x = self.pool(x)  # Max pooling to reduce dimensions
        x = self.dropout_cnn(x)  # Dropout to prevent overfitting in CNN layers

        # 2. LSTM Layers
        x = x.permute(0, 2, 1)  # Rearrange dimensions back to (batch_size, time_steps, num_features)
        lstm_out, (hn, cn) = self.lstm(x)  # Pass through LSTM layers
        x = hn[-1]  # Use the last hidden state from the LSTM for classification

        # 3. Fully Connected Layers
        x = torch.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = self.dropout_fc(x)  # Dropout to prevent overfitting in FC layers
        x = self.fc2(x)  # Output layer producing class scores
        return x


# Initialize the model with num_features and num_classes
model = CNNLSTM(num_features=num_features, num_classes=13)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, criterion, optimizer, val_loader, num_epochs=80, patience=7):
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimization

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Validation step
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_samples
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, Val Loss: {val_loss:.4f}')

        # Early stopping logic
        # if val_loss < best_loss:
        #    best_loss = val_loss
        #    torch.save(model.state_dict(), 'best_model_CNNLSTM.pt')  # Save the best model
        #    patience_counter = 0  # Reset patience counter
        # else:
        #    patience_counter += 1

        # if patience_counter >= patience:
        #    print(f'Early stopping triggered at epoch {epoch + 1}')
        #    break


# Train the model and calculate training accuracy
train_model(model, train_loader, criterion, optimizer, test_loader)
torch.save(model.state_dict(), 'latest_model_CNNLSTM_earlystopping.pt')


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


# Evaluate the model
evaluate_model(model, test_loader)


# Function to load sensor data from a CSV file for prediction
def load_sensor_data_for_prediction(file_path, time_steps, num_features):
    # Load CSV data
    data = pd.read_csv(file_path).values

    # Check if the CSV has enough data per row (num_features * time_steps columns)
    if data.shape[1] != num_features * time_steps:
        raise ValueError(f"The CSV file must have {num_features * time_steps} columns, but it has {data.shape[1]}.")

    # Reshape each row into (time_steps, num_features)
    sensor_data = data.reshape((-1, time_steps, num_features))  # Reshape each sample

    return sensor_data


# Define the predict_pose function
def predict_pose(model, sensor_data):
    model.eval()
    with torch.no_grad():
        # Ensure sensor_data is 3D as expected by the model
        if sensor_data.ndim == 3:
            sensor_data = torch.tensor(sensor_data, dtype=torch.float32)
        else:
            raise ValueError(f"Expected 3D input for sensor_data, but got {sensor_data.ndim}D input.")

        # Ensure the input shape is correct before passing to the model
        print(f"Sensor data shape: {sensor_data.shape}")

        # Pass the data to the model for prediction
        outputs = model(sensor_data)
        _, predicted = torch.max(outputs, 1)

        predicted_pose = label_encoder.inverse_transform(predicted.cpu().numpy())
        return predicted_pose


# Example usage with CSV input
csv_file_path = 'C:/Users/Malar/PycharmProjects/Gait Analysis/test/collected_data.csv'


# For real-time data prediction
def normalize_real_time_data(data, scaler, time_steps, num_features):
    # Normalize the sensor data using the scaler fitted on the training data
    data_scaled = scaler.transform(data.reshape(-1, num_features)).reshape(-1, time_steps, num_features)
    return data_scaled


# Load the sensor data from the CSV file
sensor_data = load_sensor_data_for_prediction(csv_file_path, time_steps=8,
                                              num_features=4)

sensor_data_normalized = normalize_real_time_data(sensor_data, scaler, time_steps=8, num_features=4)

# Predict the pose using the model
predicted_pose = predict_pose(model, sensor_data_normalized)

print(predicted_pose)


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
