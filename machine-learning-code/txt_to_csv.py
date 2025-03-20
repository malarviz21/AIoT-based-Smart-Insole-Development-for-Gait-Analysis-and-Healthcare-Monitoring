import os
import pandas as pd

# Define the base path where the .txt files are located
base_path = 'C:/Users/Malar/Desktop/NewData_Malar1510/latest_data/yoga13'

# List all the .txt files you want to convert
file_names = [ 'tiptoe.txt'
    #'yoga1.txt', 'yoga2.txt', 'yoga3.txt', 'yoga4.txt', 'yoga5.txt',
    #'yoga6.txt', 'yoga7.txt', 'yoga8.txt', 'yoga9.txt', 'yoga10.txt',
    #'yoga11.txt', 'yoga12.txt', 'yoga13.txt'
]

# Convert each .txt file to a .csv file
for file_name in file_names:
    txt_file_path = os.path.join(base_path, file_name)

    # Load the .txt file as a DataFrame
    try:

        df = pd.read_csv(txt_file_path, delimiter=',',
                         header=None)

        # Save the DataFrame as a .csv file
        csv_file_name = file_name.replace('.txt', '.csv')
        csv_file_path = os.path.join(base_path, csv_file_name)
        df.to_csv(csv_file_path, index=False)

        print(f"Converted {file_name} to {csv_file_name}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
