import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the base path where the CSV files are located
base_path = 'C:/Users/Malar/Desktop/Malar_datanew'  # Define the specific CSV files
csv_file_paths = [
    os.path.join(base_path, 'scope_190_2.csv'),
    os.path.join(base_path, 'scope_191_2.csv'),
    os.path.join(base_path, 'scope_192_2.csv'),
    os.path.join(base_path, 'scope_193_2.csv')
]

# Define colors for each file
colors = ['blue', 'green', 'red', 'orange']

# Create a figure for the plot
plt.figure(figsize=(12, 6))

# Loop through each CSV file and plot the points from the first two columns with different colors
for i, (csv_file_path, color) in enumerate(zip(csv_file_paths, colors)):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Plot the points from the first two columns
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], color=color, label=f'File {i + 1}')

plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Scatter Plot of Column 1 vs Column 2 for Multiple CSV Files')
plt.legend(loc='upper right')
plt.show()
