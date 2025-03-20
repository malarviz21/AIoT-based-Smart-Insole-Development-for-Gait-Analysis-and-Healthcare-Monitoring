import socket
import serial
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from sklearn.preprocessing import MinMaxScaler


# CNN-LSTM Model Definition
class CNNLSTM(nn.Module):
    def __init__(self, num_features, num_classes=13):
        super(CNNLSTM, self).__init__()
        # CNN layers with BatchNorm
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
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

        x = x.permute(0, 2, 1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # LSTM
        x = x.permute(0, 2, 1)
        lstm_out, (hn, cn) = self.lstm(x)
        x = hn[-1]  # Use the last hidden state from LSTM

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


# Function to normalize data and load it
def data_load(file_path, scaler):
    columns_to_use = list(range(1, 32))
    csv_data = pd.read_csv(file_path, usecols=columns_to_use)

    # Fit the scaler on the collected data
    scaler.fit(csv_data)

    normalized_data = scaler.transform(csv_data)

    return normalized_data


# Function to handle serial data collection and model prediction
def read_and_send_data_from_serial(port, tcp_port, host, model, scaler, num_features, time_steps):
    ser = serial.Serial(port, 115200, timeout=1)  # Open the serial connection
    collected_data = []

    # Set up socket communication
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, tcp_port))
        print(f"Connected to {host}:{tcp_port} from {port}")

        try:
            while True:
                if len(collected_data) < 6000:
                    signal = ser.readline().decode('utf-8', errors='ignore').strip().split('\t')  # Read serial data
                    for row in signal:
                        # Remove empty values and keep only valid numbers
                        row = row.strip(',')  # Strip leading/trailing commas
                        row_list = [x for x in row.split(',') if x]  # Remove empty elements

                        cleaned_row = row_list[:32]

                    if len(cleaned_row) == 32:  # Ensure the signal has 32 elements
                        try:
                            float_signal = []
                            for item in cleaned_row:
                                try:
                                    float_signal.append(float(item))
                                except ValueError:
                                    print(f"Skipping invalid value: {item}")

                            # Ensure you still have 32 valid floats after conversion
                            if len(float_signal) == 32:
                                try:
                                    collected_data.append(float_signal)  # Collect the data

                                    print(collected_data)
                                except ValueError:
                                    print("Invalid input. Please enter a valid list with 32 elements.")
                        except Exception as e:
                            print(f"Error from {port}: {e}")
                else:
                    if len(collected_data) >= 6000:  # When enough data is collected
                        print("6000 rows collected, saving to CSV...")
                        # Save the data to CSV
                        with open(f"./test/collected_data.csv", "w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerows(collected_data)

                    # Load and preprocess data
                    data = data_load('./test/collected_data.csv', scaler)
                    # Calculate the total required size for reshaping
                    required_size = (len(data) // (time_steps * num_features)) * time_steps * num_features

                    # Truncate or reshape the data
                    data = data[:required_size]
                    reshaped_data = data.reshape(-1, time_steps, num_features).astype(np.float32)
                    tensor_data = torch.from_numpy(reshaped_data)

                    # Get prediction from the model
                    with torch.no_grad():
                        outputs = model(tensor_data)
                        _, predicted = torch.max(outputs.data, 1)

                    print(predicted.shape)
                    print(predicted)
                    # Send the prediction to the TCP server
                    incremented_tensor = predicted[0] + 1
                    str_value = str(incremented_tensor.item())
                    s.sendall(str_value.encode())
                    print(f"Sent prediction to {host}:{tcp_port}")

                    collected_data = []  # Reset collected data after sending
        except KeyboardInterrupt:
            print(f"Transmission stopped from {port}.")
        except Exception as e:
            print(f"Error from {port}: {e}")


# Main function
if __name__ == '__main__':
    # Load scaler
    scaler = MinMaxScaler()

    # Initialize the CNN-LSTM model
    num_features = 4
    time_steps = 8
    model_path = 'latest_model_CNNLSTM_new.pt'
    model = CNNLSTM(num_features=num_features, num_classes=13)
    model.load_state_dict(torch.load(model_path))

    # Start thread to read from the serial port and send data via TCP
    threading.Thread(target=read_and_send_data_from_serial,
                     args=('COM5', 65432, '127.0.0.1', model, scaler, num_features, time_steps)).start()
