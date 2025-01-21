import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

def plot_kalman_results(data_file):
    data = load_data(data_file)
    
    # Assuming columns are: time, measurement_x, measurement_y, filtered_x, filtered_y
    time = data[:, 0]
    measurements = data[:, 1:3]
    filtered = data[:, 3:5]
    
    plt.figure(figsize=(12, 8))
    
    # Plot measurements
    plt.scatter(measurements[:, 0], measurements[:, 1], 
               color='red', alpha=0.5, label='Measurements', 
               marker='x', s=100)
    
    # Plot filtered path
    plt.plot(filtered[:, 0], filtered[:, 1], 
            color='blue', linewidth=2, label='Kalman Filter')
    
    plt.title('Kalman Filter Tracking Results')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    # Make axis equal for better visualization
    plt.axis('equal')
    
    # Save the plot
    plt.savefig('kalman_results.png')
    plt.show()

if __name__ == "__main__":
    data_file = "kalman_output.csv"
    if Path(data_file).exists():
        plot_kalman_results(data_file)
    else:
        print(f"Error: {data_file} not found") 