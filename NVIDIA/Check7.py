import subprocess
import cv2
import numpy as np
import time


def get_gpu_memory():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, text=True)
        memory_info = result.stdout.strip().split('\n')
        used_memory = []
        for info in memory_info:
            used, total = map(int, info.split(', '))
            used_memory.append((used / total) * 100)  # Memory usage in percentage
        return used_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return []


def plot_memory_usage_cv2(time_intervals, memory_values):
    # Create a blank image
    height, width = 500, 800
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw the plot
    max_mem = max(memory_values) if memory_values else 100
    min_mem = min(memory_values) if memory_values else 0

    if max_mem == min_mem:
        max_mem += 1

    # Scale values to fit in the image
    scaled_values = [int((val - min_mem) / (max_mem - min_mem) * (height - 50)) for val in memory_values]

    # Draw axes
    cv2.line(img, (50, 30), (50, height - 30), (0, 0, 0), 2)  # Y-axis
    cv2.line(img, (50, height - 30), (width - 30, height - 30), (0, 0, 0), 2)  # X-axis

    # Draw labels
    cv2.putText(img, 'Memory Usage (%)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, 'Time (s)', (width - 100, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Draw the graph
    for i in range(1, len(scaled_values)):
        cv2.line(img, (50 + (i - 1) * (width - 80) // len(scaled_values), height - 30 - scaled_values[i - 1]),
                 (50 + i * (width - 80) // len(scaled_values), height - 30 - scaled_values[i]),
                 (0, 0, 255), 2)

    # Show the image
    cv2.imshow('GPU Memory Usage', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    interval = 1  # Interval in seconds
    duration = 10  # Total duration in seconds
    num_points = duration // interval

    time_intervals = []
    memory_values = []

    start_time = time.time()

    while len(time_intervals) < num_points:
        # Collect memory data
        memory_data = get_gpu_memory()
        if memory_data:
            avg_memory_usage = sum(memory_data) / len(memory_data)  # Average usage of all GPUs
            memory_values.append(avg_memory_usage)
            elapsed_time = time.time() - start_time
            time_intervals.append(elapsed_time)
        else:
            memory_values.append(0)
            elapsed_time = time.time() - start_time
            time_intervals.append(elapsed_time)

        # Wait for the next interval
        # time.sleep(interval)

    # Plotting the memory usage using OpenCV
    plot_memory_usage_cv2(time_intervals, memory_values)


if __name__ == "__main__":
    main()
