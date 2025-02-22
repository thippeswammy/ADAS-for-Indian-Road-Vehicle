import time

import matplotlib.pyplot as plt
import psutil
from pynvml import *


# Function to initialize GPU monitoring
def initialize_gpu():
    try:
        nvmlInit()
        return True
    except NVMLError as err:
        print(f"Error initializing GPU monitoring: {err}")
        return False


# Function to monitor CPU, RAM, and GPU
def monitor_system():
    # Monitor CPU and RAM
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent

    # Monitor GPU
    gpu_usage_dedicated = 0
    gpu_usage_shared = 0
    if initialize_gpu():
        handle = nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        gpu_usage_dedicated = memory_info.used / 1024 ** 2  # in MB
        # Get shared memory usage (this may vary depending on the specific GPU and driver)
        total_memory = memory_info.total / 1024 ** 2  # in MB
        gpu_usage_shared = total_memory - gpu_usage_dedicated  # Assuming shared memory is the remaining memory

    # Calculate Total GPU memory usage
    total_gpu_usage = gpu_usage_dedicated + gpu_usage_shared

    return cpu_usage, ram_usage, gpu_usage_dedicated, gpu_usage_shared, total_gpu_usage


# Function to start collecting data based on True/False input
def collect_data():
    system_parameters = []
    print("Monitoring started...")

    while collect_flag:  # Only collect when collect_flag is True
        cpu_usage, ram_usage, gpu_usage_dedicated, gpu_usage_shared, total_gpu_usage = monitor_system()
        # Record the data
        system_parameters.append((cpu_usage, ram_usage, gpu_usage_dedicated, gpu_usage_shared, total_gpu_usage))

        time.sleep(1)  # Pause before collecting the next data point

    # Return the collected data for plotting
    return system_parameters


# Function to plot and save the results
def plot_and_save_results(system_parameters, width, height, output_dir):
    plt.figure(figsize=(12, 6))

    # Extract CPU, RAM, and GPU usage from the collected data
    cpu_usages = [p[0] for p in system_parameters]
    ram_usages = [p[1] for p in system_parameters]
    gpu_usages_dedicated = [p[2] for p in system_parameters]
    gpu_usages_shared = [p[3] for p in system_parameters]
    total_gpu_usages = [p[4] for p in system_parameters]

    # Plot the data with specific markers
    plt.plot(cpu_usages, label='CPU Usage (%)', marker='o')
    plt.plot(ram_usages, label='RAM Usage (%)', marker='o')
    plt.plot(gpu_usages_dedicated, label='GPU Dedicated Memory Usage (MB)', marker='o')
    plt.plot(gpu_usages_shared, label='GPU Shared Memory Usage (MB)', marker='o')
    plt.plot(total_gpu_usages, label='Total GPU Memory Usage (MB)', marker='o', linestyle='--')

    # Add title, labels, and grid
    plt.title(f'System Usage for Resolution {width}x{height}')
    plt.xlabel('Time')
    plt.ylabel('Usage (%) or Memory (MB)')
    plt.legend()
    plt.grid(True)

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'system_usage'), exist_ok=True)

    # Save the plot to a PNG file
    plt.savefig(os.path.join(output_dir, 'system_usage', f'system_usage_{width}x{height}.png'), format='png')
    plt.close()


# Function to control monitoring and plotting
def start():
    global collect_flag, system_parameters, thread  # Make collect_flag and system_parameters global to modify them within the main function
    # Set the resolution for the title and filename
    width, height = 1920, 1080

    # Start collecting data in a separate thread
    collect_flag = True
    system_parameters = []  # Initialize system_parameters here

    def start_monitoring():
        global system_parameters  # Use global to modify system_parameters within the function
        system_parameters = collect_data()  # Collect data until collect_flag is False

    # Create a new thread for data collection
    thread = threading.Thread(target=start_monitoring)
    thread.start()


def stop(output_dir='./output'):
    global collect_flag, thread, system_parameters  # Make collect_flag, thread, and system_parameters global to modify them within the main function
    # Set the resolution for the title and filename
    width, height = 1920, 1080
    # Collect data for a specific amount of time (e.g., 10 seconds)
    collect_flag = False  # Stop the collection process

    # Wait for the thread to finish
    thread.join()

    # Plot and save the results once data collection is stopped
    plot_and_save_results(system_parameters, width, height, output_dir)


# Call the main function
start()

time.sleep(3)
stop()
