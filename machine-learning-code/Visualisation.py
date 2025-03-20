import scipy.io as sio
import matplotlib.pyplot as plt
import os


base_path = 'C:/Users/Malar/Desktop/Malar data - 1510'

# Define file names
file_names = [
    'jump.mat',
    'run.mat',
    'tiptoe.mat',
    'walk.mat',
    'slip.mat',
    'yoga1.mat',
    'yoga2.mat',
    'yoga3.mat',
    'yoga4.mat',
    'yoga5.mat',
    'yoga6.mat',
    'yoga7.mat',
    'yoga8.mat'
]

new_file_name = ['Yoga 2.mat']


# Function to load and plot data from a .mat file
def plot_data_from_mat(file_path, key):
    # Load the .mat file
    data = sio.loadmat(file_path)

    # Extract the data using the key
    data_array = data[key]

    # Plot the data (time index from 0 to 1500)
    plt.figure()
    for i in range(data_array.shape[1]):
        plt.plot(data_array[0:1500, i])
    plt.title(f'Data from {key} (Time index 0 to 1500)')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    # Save the plot
    plot_file_path = os.path.join(base_path, f'{key}_plot.png')
    plt.savefig(plot_file_path)
    plt.close()


# Loop through each .mat file and plot the data
for file_name in new_file_name:
    file_path = os.path.join(base_path, file_name)
    key = file_name.replace('.mat', '')
    plot_data_from_mat(file_path, key)
