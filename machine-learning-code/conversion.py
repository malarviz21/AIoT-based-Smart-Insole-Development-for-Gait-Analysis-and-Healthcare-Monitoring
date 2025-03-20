import numpy as np
import scipy.io as sio
import os


# Define a function to convert a text file with comma-separated values to a numpy array
def txt_to_array(file_path):
    return np.loadtxt(file_path, delimiter=',')


# Define the base path where the files are located
base_path = 'C:/Users/Malar/Desktop/Malar data - 1510'

# Define file paths
file_names = [
    'jump.txt',
    'run.txt',
    'tiptoe.txt',
    'walk.txt',
    'slip.txt',
    'yoga1.txt',
    'yoga2.txt',
    'yoga3.txt',
    'yoga4.txt',
    'yoga5.txt',
    'yoga6.txt',
    'yoga7.txt',
    'yoga8.txt'
]

new_file_names = ['Yoga 2.txt']

# Convert each text file to a numpy array and save to a separate .mat file
for file_name in new_file_names:
    file_path = os.path.join(base_path, file_name)
    key = file_name.replace('.txt', '')
    data_array = txt_to_array(file_path)
    output_path = os.path.join(base_path, f'{key}.mat')
    sio.savemat(output_path, {key: data_array})
    print(f"Data from {file_path} saved to {output_path}")
