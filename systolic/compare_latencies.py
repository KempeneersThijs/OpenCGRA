import matplotlib.pyplot as plt
import numpy as np

def read_latencies(file_path):
    """
    Reads latencies from a file where each line contains latency values for different workloads.

    Parameters:
    - file_path (str): Path to the file containing latencies.

    Returns:
    - latencies (list of float): List of latencies for each workload.
    """
    latencies = []
    with open(file_path, 'r') as file:
        for line in file:
            if "execution latency:" in line:
                try:
                    # Extract the latency value after 'execution latency:'
                    latency = float(line.split("execution latency:  ")[1].split(" cycles")[0].strip())
                    latencies.append(latency)
                except ValueError:
                    continue  # Skip lines that cannot be converted to float
    return latencies

def create_comparison_bar_chart(workloads, latencies_a, latencies_b, output_file):
    """
    Creates a bar chart comparing latencies of two frameworks for given workloads.

    Parameters:
    - workloads (list of str): List of workload names.
    - latencies_a (list of float): List of latencies for Framework A.
    - latencies_b (list of float): List of latencies for Framework B.
    - output_file (str): Path to save the bar chart image.
    """
    num_workloads = len(workloads)

    # Define bar width and positions
    bar_width = 0.35
    indices = np.arange(num_workloads)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each framework
    bars_a = ax.bar(indices - bar_width/2, latencies_a, bar_width, label='Framework A', color='skyblue')
    bars_b = ax.bar(indices + bar_width/2, latencies_b, bar_width, label='Framework B', color='lightgreen')

    # Add latency numbers on top of the bars
    for bar in bars_a:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
    
    for bar in bars_b:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom')
    
    # Set x-axis labels and positions
    ax.set_xlabel('Workloads')
    ax.set_ylabel('Execution Latency (ms)')
    ax.set_title('Execution Latency Comparison Between Frameworks')
    ax.set_xticks(indices)
    ax.set_xticklabels(workloads)

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle='--', linewidth=0.5)

    # Save the bar chart to a file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# File containing latency numbers for Framework A
input_file_path = 'systolic2x2/output_all.txt'

# Read latencies for Framework A from the file
latencies_a = read_latencies(input_file_path)

# Latencies for Framework B provided by the user
latencies_b = [8.5, 9.2, 7.8, 6.4, 8.1, 7.0, 8.0, 8.0]  # Example user input latencies

# Workloads names
workloads = [f'Workload {i+1}' for i in range(len(latencies_b))]

# Output file for the bar chart
output_file_path = 'latency_comparison_chart.png'

# Create and save the bar chart
create_comparison_bar_chart(workloads, latencies_a, latencies_b, output_file_path)

print(f"Bar chart saved as {output_file_path}")
