import subprocess
import time
from collections import deque

import GPUtil
import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pyopencl as cl
import tensorflow as tf
import torch


def check_cuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using NVIDIA GPU')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using CPU')
    return device


def check_amd_gpu():
    try:
        output = subprocess.check_output(['clinfo'], universal_newlines=True)
        return 'AMD' in output
    except subprocess.CalledProcessError:
        return False


def display_amd_gpu_info():
    if check_amd_gpu():
        print("AMD GPU is present.")
    else:
        print("No AMD GPU found.")


def gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.memoryTotal} MB")
        print(f"GPU Memory Used: {gpu.memoryUsed} MB")
        print(f"GPU Memory Free: {gpu.memoryFree} MB")
        print(f"GPU Memory Utilization: {gpu.memoryUtil * 100}%")


def get_system_memory_info():
    virtual_memory = psutil.virtual_memory()
    print(f"Total system memory: {virtual_memory.total / (1024 * 1024)} MB")
    print(f"Available memory: {virtual_memory.available / (1024 * 1024)} MB")
    print(f"Used memory: {virtual_memory.used / (1024 * 1024)} MB")
    print(f"Memory usage percentage: {virtual_memory.percent}%")
    print("-" * 50)
    swap_memory = psutil.swap_memory()
    print(f"Total swap memory: {swap_memory.total / (1024 * 1024)} MB")
    print(f"Used swap memory: {swap_memory.used / (1024 * 1024)} MB")
    print(f"Free swap memory: {swap_memory.free / (1024 * 1024)} MB")
    print(f"Swap memory usage percentage: {swap_memory.percent}%")


def get_amd_gpu_memory_info():
    platforms = cl.get_platforms()
    for platform in platforms:
        if 'AMD' in platform.vendor or 'amd' in platform.vendor:
            devices = platform.get_devices()
            for device in devices:
                if cl.device_type.to_string(device.type) == 'GPU':
                    print(f"Device: {device.name}")
                    print(f"Global Memory Size: {device.global_mem_size / (1024 * 1024)} MB")
                    print(f"Local Memory Size: {device.local_mem_size / 1024} KB")
                    print(f"Max Allocable Memory Size: {device.max_mem_alloc_size / (1024 * 1024)} MB")
                    print(f"Max Work Group Size: {device.max_work_group_size}")
                    print()


def check_tensorflow_cuda():
    print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())
    devices = tf.config.list_physical_devices()
    print("Available devices:")
    for device in devices:
        print(device)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("TensorFlow can access the following GPU(s):")
        for gpu in gpus:
            print(gpu)
    else:
        print("No GPU devices accessible to TensorFlow.")


def get_gpu_names():
    devices = tf.config.list_physical_devices('GPU')
    if not devices:
        print("No GPU devices found.")
        return
    print("Found GPU devices:")
    for i, device in enumerate(devices):
        device_name = device.name
        device_details = tf.config.experimental.get_device_details(device)
        print(f"GPU {i}: {device_name}")
        for key, value in device_details.items():
            print(f"  {key}: {value}")
        print("=" * 50)


def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, text=True, check=True)
        memory_info = result.stdout.strip().split('\n')
        used_memory = []
        total_memory = []
        for info in memory_info:
            used, total = map(int, info.split(', '))
            used_memory.append(used)
            total_memory.append(total)
        return used_memory, total_memory
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        return [], []


def update_graph(i, gpu_ids, lines, ax, history):
    used_memory, total_memory = get_gpu_memory()
    if not used_memory or not total_memory:
        print("Error: No GPU memory data retrieved.")
        return
    current_time = time.time()
    history['time'].append(current_time)
    history['used_memory'].append(used_memory)
    history['total_memory'].append(total_memory)
    if len(history['time']) > 20:
        history['time'].popleft()
        history['used_memory'].popleft()
        history['total_memory'].popleft()
    if len(history['time']) < 2:
        print("Not enough data to plot.")
        return
    for j, line in enumerate(lines):
        if j < len(used_memory):
            gpu_used_memory = [memory[j] for memory in history['used_memory']]
            gpu_total_memory = [memory[j] for memory in history['total_memory']]
            memory_percentage = [used / total * 100 if total > 0 else 0 for used, total in
                                 zip(gpu_used_memory, gpu_total_memory)]
            line.set_xdata(list(history['time']))
            line.set_ydata(memory_percentage)
    ax.relim()
    ax.autoscale_view()
    if len(history['time']) > 1:
        ax.set_xlim(min(history['time']), max(history['time']))
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")


def plot_gpu_memory_usage():
    gpu_ids = list(range(len(get_gpu_memory()[0])))
    fig, ax = plt.subplots()
    lines = []
    for gpu_id in gpu_ids:
        line, = ax.plot([], [], label=f"GPU {gpu_id}")
        lines.append(line)
    ax.set_xlim(time.time() - 20, time.time())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory Usage (%)')
    ax.set_title('GPU Memory Usage Tracker')
    history = {
        'time': deque(maxlen=20),
        'used_memory': deque(maxlen=20),
        'total_memory': deque(maxlen=20)
    }
    ani = animation.FuncAnimation(fig, update_graph, fargs=(gpu_ids, lines, ax, history),
                                  interval=1000, blit=False, cache_frame_data=False)
    plt.show()


def plot_memory_usage_cv2(time_intervals, memory_values):
    height, width = 500, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    max_mem = max(memory_values) if memory_values else 100
    min_mem = min(memory_values) if memory_values else 0
    if max_mem == min_mem:
        max_mem += 1
    scaled_values = [int((val - min_mem) / (max_mem - min_mem) * (height - 50)) for val in memory_values]
    cv2.line(img, (50, 30), (50, height - 30), (0, 0, 0), 2)
    cv2.line(img, (50, height - 30), (width - 30, height - 30), (0, 0, 0), 2)
    cv2.putText(img, 'Memory Usage (%)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'Time (s)', (width - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for i in range(1, len(scaled_values)):
        cv2.line(img, (50 + (i - 1) * (width - 80) // len(scaled_values), height - 30 - scaled_values[i - 1]),
                 (50 + i * (width - 80) // len(scaled_values), height - 30 - scaled_values[i]),
                 (0, 0, 255), 2)
    cv2.imshow('GPU Memory Usage', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main_cv2():
    interval = 1
    duration = 10
    num_points = duration // interval
    time_intervals = []
    memory_values = []
    start_time = time.time()
    while len(time_intervals) < num_points:
        memory_data = get_gpu_memory()[0]
        if memory_data:
            avg_memory_usage = sum(memory_data) / len(memory_data)
            memory_values.append(avg_memory_usage)
            elapsed_time = time.time() - start_time
            time_intervals.append(elapsed_time)
        else:
            memory_values.append(0)
            elapsed_time = time.time() - start_time
            time_intervals.append(elapsed_time)
        time.sleep(interval)
    plot_memory_usage_cv2(time_intervals, memory_values)


if __name__ == "__main__":
    device = check_cuda()
    display_amd_gpu_info()
    gpu_memory_usage()
    get_system_memory_info()
    get_amd_gpu_memory_info()
    check_tensorflow_cuda()
    get_gpu_names()
    plot_gpu_memory_usage()
    main_cv2()
