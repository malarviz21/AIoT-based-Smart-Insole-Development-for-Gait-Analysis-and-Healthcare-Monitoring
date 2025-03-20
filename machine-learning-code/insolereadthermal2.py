import tkinter as tk
from tkinter import ttk
import numpy as np
import serial
import PIL.Image
from scipy.interpolate import griddata
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Replace with your actual port and baud rate
Arduino = serial.Serial("COM5", 115200)

channelNumber = 32

# Create Tkinter window
root = tk.Tk()
root.title("Heatmap")
root.geometry("800x500")

# Create a figure and a canvas for matplotlib
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Load the background image
png_file_path = 'C:/Users/Malar/Desktop/foot2.png'  # Change the path to your PNG image
png_img = PIL.Image.open(png_file_path)
png_img = png_img.convert('RGBA')
png_img = png_img.rotate(180) 
background_array = np.array(png_img)

# Specific coordinates
rx = 7004.5588 / 266.487462
specific_coordinates = [(6490.3183 / rx, 1916.4496 / rx), (5387.6680 / rx, 7578.0397 / rx), (3840.0019 / rx, 3553.1189 / rx), (4838.8151 / rx, 1929.3388 / rx),
                        (2320.0353 / rx, 16570.2409 / rx), (4621.2519 / rx, 13625.3103 / rx), (5257.1397 / rx, 18765.6523 / rx), (7779.9044 / rx, 14781.1069 / rx),
                        (7004.5588 / rx, 6964.9069 / rx), (6777.1062 / rx, 10465.6980 / rx), (3108.1981 / rx, 8408.7360 / rx), (2574.1791 / rx, 12972.6205 / rx),
                        (2755.1618 / rx, 19893.0258 / rx), (5197.8043 / rx, 16965.8106 / rx), (5217.5827 / rx, 22454.3391 / rx), (8134.9086 / rx, 18528.3106 / rx),
                        (7049.0604 / rx, 5323.2930 / rx), (5367.8895 / rx, 10035.5160 / rx), (3216.9797 / rx, 5254.0683 / rx), (3330.8517 / rx, 1999.2298 / rx),
                        (2171.6966 / rx, 18122.8517 / rx), (5033.1678 / rx, 15258.2628 / rx), (5267.0290 / rx, 20694.0542 / rx), (7976.6809 / rx, 20199.5923 / rx),
                        (6322.2013 / rx, 3656.9559 / rx), (5170.1047 / rx, 5313.4038 / rx), (3068.6410 / rx, 10851.3783 / rx), (2331.8927 / rx, 14816.9638 / rx),
                        (3289.1808 / rx, 21841.2061 / rx), (6361.7582 / rx, 13343.4669 / rx), (7304.2125 / rx, 21465.4151 / rx), (7986.5700 / rx, 16698.8010 / rx)]

Pre_speed = 0
dataset1 = []

def canFloat(data):
    try:
        float(data)
        return True
    except:
        return False

def dataProcess(data):
    data = str(data)
    dataSet = []
    datapoint = ''
    for i in data:
        if i.isdigit() or i == '.':
            datapoint += i
        elif i == "," and canFloat(datapoint):
            dataSet.append(float(datapoint))
            datapoint = ''
    if len(dataSet) == channelNumber:  # Ensure the signal has exactly 32 data points
        return dataSet
    else:
        return None

def plotData():
    global dataset1
    signal = Arduino.readline()
    signal = dataProcess(signal)
    if signal:
        #print("Received Signal:", signal)  # Debug print statement
        dataset1.append(signal)

        # Prepare data for interpolation
        X = np.array(specific_coordinates)
        y = np.array(signal)

        # Create grid for interpolation
        grid_x, grid_y = np.mgrid[0:background_array.shape[1]:100j, 0:background_array.shape[0]:100j]
        
        # Interpolate data
        grid_z = griddata(X, y, (grid_x, grid_y), method='cubic')
        
        # Clear the previous plot
        ax.clear()
        
        # Display the background image
        ax.imshow(background_array, extent=[0, background_array.shape[1], 0, background_array.shape[0]], origin='upper')
        
        # Overlay the heatmap
        #heatmap = ax.imshow(grid_z.T, extent=[0, background_array.shape[1], 0, background_array.shape[0]], origin='lower', cmap='jet', alpha=0.5)
        # Overlay the heatmap with vmin and vmax set to highlight the range
        heatmap = ax.imshow(grid_z.T, extent=[0, background_array.shape[1], 0, background_array.shape[0]], origin='lower', cmap='jet', alpha=0.5, vmin=-5, vmax=80)

        # Update the canvas
        canvas.draw()
    else:
        print("Invalid Signal Received")  # Debug print statement

# Set up a timer to periodically call plotData
def update():
    plotData()
    root.after(1, update)

# Start the update loop
update()

# Start Tkinter main loop
root.mainloop()

# Save the dataset ensuring all rows have the same length
padded_dataset = np.array(dataset1)
np.savetxt("C:/Users/Malar/Desktop/yoga1.txt", padded_dataset, delimiter=",")
