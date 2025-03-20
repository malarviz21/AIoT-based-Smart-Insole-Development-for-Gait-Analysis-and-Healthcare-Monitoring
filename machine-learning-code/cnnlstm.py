import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Set up scaler and device
scaler = MinMaxScaler()
device = 'cpu'
datas = []
labels = []

base_path = 'C:/Users/Malar/Desktop/NewData_Malar1510/finaldata'

# Load data from .txt files 
for label in range(1, 14):  # 13 yoga poses
    # folder_path = os.path.join(base_path, 'yoga' + str(label)) 
    for file in os.listdir(base_path):
        file_path = os.path.join(base_path, file)

        # Load the .txt file as a dataframe
        txt_data = pd.read_csv(file_path, delimiter=',', header=None) 
        # Handle NaN values (drop or fill them)
        txt_data = np.nan_to_num(txt_data, nan=0.0)

        normalized_data = scaler.fit_transform(txt_data)

        # Add normalized data and label
        datas.append(normalized_data)

        labels.append(label - 1)  



# Function to convert label to one-hot encoding
def to_one_hot(label, num_classes=13): 
    one_hot = np.zeros(num_classes)
    one_hot[label] = 1
    return one_hot


one_hot_labels = [to_one_hot(label) for label in labels]
print(one_hot_labels)

# Determine max sequence length and pad data
padding_datas = []
max_length = 0
for data in datas:
    max_length = max(max_length, len(data))
    padding_datas.append(data[:, :32])  

print(max_length)

# Padding each data sequence to the max length
def pad_to_shape(arr, target_shape):
    pad_shape = ((0, target_shape[0] - arr.shape[0]), (0, target_shape[1] - arr.shape[1]))
    return np.pad(arr, pad_shape, mode='constant')


padded_data_list = [pad_to_shape(data, target_shape=(max_length, 32)) for data in padding_datas]
print(len(padded_data_list), len(padded_data_list[0]))


# Split data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(padded_data_list, labels, test_size=0.2,
                                                                    random_state=42)


# Define Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = torch.tensor(self.data[idx], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[idx], dtype=torch.long)  
        return sample_data, sample_label


# Create datasets and data loaders
train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# CNN-LSTM Hybrid Model Definition
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=13):
        super(CNNLSTM, self).__init__()

        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.2)

        # LSTM layers
        self.lstm = nn.LSTM(input_size=64, hidden_size=100, num_layers=2, batch_first=True, dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, num_classes)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        # CNN
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # LSTM
        x = x.permute(0, 2, 1)  # Reshape for LSTM input (batch, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        x = hn[-1]  # Use the last hidden state from LSTM

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


# Hyperparameters
input_size = 32  # 32 channels
hidden_size = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
num_classes = 13  # Number of classes (yoga postures)
dropout = 0.3  # Dropout rate for regularization

# Instantiate the model
#model = CNNLSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes,
#                     dropout=dropout)
model = CNNLSTM(num_classes=num_classes)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000


# Training function
def train(model, save_model_path):
    best_acc = 0
    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.view(-1, max_length, 32).to(device)  
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation and model saving
        if (epoch + 1) % 10 == 0:
            val_acc = test(model)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation acc: {val_acc:.2f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_model_path)


# Testing function
def test(model, model_path=None):
    if model_path:
        model.load_state_dict(torch.load(model_path))
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.view(-1, max_length, 32).to(device)  
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels.to(device)).sum().item()

    acc = 100 * correct / total
    print(f"Accuracy: {acc}%")
    return acc

# To train the model
train(model, './model_CNNLSTM.pt')

# To test the model
# test(model, 'model_CNNLSTM.pt')
