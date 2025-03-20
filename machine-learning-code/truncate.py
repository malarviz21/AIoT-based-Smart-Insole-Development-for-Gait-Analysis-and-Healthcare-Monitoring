import numpy as np
import scipy.io as sio
import os


# Define a function to convert a text file with comma-separated values to a numpy array
def txt_to_array(file_path, target_columns=32):
    try:
        data = np.loadtxt(file_path, delimiter=',')
        print(f"Loaded data from {file_path}, shape: {data.shape}")

        # Ensure the data has the correct number of columns
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] > target_columns:
            data = data[:, :target_columns]  # Truncate to 32 columns if more

        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Define the base path where the files are located
base_path = 'C:/Users/Malar/Desktop/Malar data - 1510'

# Define file paths
new_file_names = ['Yoga 2.txt']

# Convert each text file to a numpy array and save to a separate .mat file
for file_name in new_file_names:
    file_path = os.path.join(base_path, file_name)
    key = file_name.replace('.txt', '')
    data_array = txt_to_array(file_path, target_columns=32)  # Ensure truncation to 32 columns

    if data_array is not None:
        output_path = os.path.join(base_path, f'{key}.mat')
        sio.savemat(output_path, {key: data_array})
        print(f"Data from {file_path} saved to {output_path}")
