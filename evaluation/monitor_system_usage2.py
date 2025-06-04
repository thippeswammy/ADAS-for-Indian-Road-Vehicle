import time

import matplotlib.pyplot as plt
import psutil
from pynvml import *

width, height = 100, 100
total_ram, total_gpu_memory = 0


# Function to initialize GPU monitoring
def initialize_gpu():
    try:
        nvmlInit()
        return True
    except NVMLError as err:
        print(f"Error initializing GPU monitoring: {err}")
        return False


# Function to monitor CPU, RAM, and GPU
# Function to monitor CPU, RAM, and GPU
def monitor_system():
    global total_ram, total_gpu_memory
    # Monitor CPU and RAM
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_info = psutil.virtual_memory()
    ram_usage = ram_info.percent
    total_ram = ram_info.total / 1024 ** 2  # in MB

    # Monitor GPU
    gpu_usage_dedicated = 0
    total_gpu_memory = 0
    gpu_usage_percentage = 0
    if initialize_gpu():
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        total_gpu_memory = memory_info.total / 1024 ** 2  # in MB
        gpu_usage_dedicated = memory_info.used / 1024 ** 2  # in MB
        gpu_usage_percentage = (gpu_usage_dedicated / total_gpu_memory) * 100

    # Return usage percentages and total memory values
    return cpu_usage, ram_usage, gpu_usage_percentage,


# Function to start collecting data based on True/False input
def collect_data():
    system_parameters = []
    print("Monitoring started...")

    while collect_flag:  # Only collect when collect_flag is True
        cpu_usage, ram_usage, gpu_usage_percentage = monitor_system()
        # Record the data
        system_parameters.append((cpu_usage, ram_usage, gpu_usage_percentage))

        time.sleep(1)  # Pause before collecting the next data point

    # Return the collected data for plotting
    return system_parameters


# Function to plot and save the results
def plot_and_save_results(system_parameters, width, height, output_dir):
    plt.figure(figsize=(12, 6))

    # Extract CPU, RAM, and GPU usage from the collected data
    cpu_usages = [p[0] for p in system_parameters]
    ram_usages = [p[1] for p in system_parameters]
    gpu_usages = [p[2] for p in system_parameters]

    # Plot the data with specific markers
    plt.plot(cpu_usages, label='CPU Usage (%)', marker='o')
    plt.plot(ram_usages, label='RAM Usage (%)', marker='o')
    plt.plot(gpu_usages, label='GPU Usage (%)', marker='o')

    # Add title, labels, and grid
    plt.title(f'System Usage for Resolution {width}x{height}')
    plt.xlabel('Time')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True)

    # Display total GPU and RAM on the plot
    text = f"Total RAM: {total_ram:.2f} MB\nTotal GPU Memory: {total_gpu_memory:.2f} MB"
    plt.annotate(text, xy=(0.02, 0.95), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"))
    # Save the plot to a PNG file
    plt.savefig(output_dir, format='png')
    plt.close()


def start(width_1, height_1):
    global collect_flag, system_parameters, thread, width, height  # Make collect_flag and system_parameters global to modify them within the main function
    # Set the resolution for the title and filename
    width, height = width_1, height_1

    # Start collecting data in a separate thread
    collect_flag = True
    system_parameters = []  # Initialize system_parameters here

    def start_monitoring():
        global system_parameters  # Use global to modify system_parameters within the function
        system_parameters = collect_data()  # Collect data until collect_flag is False

    # Create a new thread for data collection
    thread = threading.Thread(target=start_monitoring)
    thread.start()


def stop(output_dir='./output/img.png'):
    global collect_flag, thread, system_parameters, width, height  # Make collect_flag, thread, and system_parameters global to modify them within the main function
    # Set the resolution for the title and filename
    # Collect data for a specific amount of time (e.g., 10 seconds)
    collect_flag = False  # Stop the collection process

    # Wait for the thread to finish
    thread.join()
    # Plot and save the results once data collection is stopped
    plot_and_save_results(system_parameters, width, height, output_dir)
