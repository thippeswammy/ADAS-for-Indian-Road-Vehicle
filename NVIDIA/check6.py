import subprocess
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import deque

# Number of recent values to keep
RECENT_VALUES_COUNT = 20
UPDATE_INTERVAL = 1  # Interval in seconds


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

    # Trim the history to the most recent values
    if len(history['time']) > RECENT_VALUES_COUNT:
        history['time'].popleft()
        history['used_memory'].popleft()
        history['total_memory'].popleft()

    # Ensure there's enough data to plot
    if len(history['time']) < 2:
        print("Not enough data to plot.")
        return

    # Debugging: print lengths of historical data
    print(f"Time Length: {len(history['time'])}")
    for j, line in enumerate(lines):
        if j < len(used_memory):
            gpu_used_memory = [memory[j] for memory in history['used_memory']]
            gpu_total_memory = [memory[j] for memory in history['total_memory']]
            memory_percentage = [used / total * 100 if total > 0 else 0 for used, total in
                                 zip(gpu_used_memory, gpu_total_memory)]

            # Debugging: print the last few data points
            print(f"GPU {j} - Data Points: {list(zip(history['time'], memory_percentage))}")

            line.set_xdata(list(history['time']))
            line.set_ydata(memory_percentage)

    # Ensure the x and y limits are updated
    ax.relim()
    ax.autoscale_view()

    # Adjust x-axis limits based on the range of data
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

    ax.set_xlim(time.time() - RECENT_VALUES_COUNT, time.time())
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory Usage (%)')
    ax.set_title('GPU Memory Usage Tracker')

    history = {
        'time': deque(maxlen=RECENT_VALUES_COUNT),
        'used_memory': deque(maxlen=RECENT_VALUES_COUNT),
        'total_memory': deque(maxlen=RECENT_VALUES_COUNT)
    }

    ani = animation.FuncAnimation(fig, update_graph, fargs=(gpu_ids, lines, ax, history),
                                  interval=UPDATE_INTERVAL * 1000, blit=False, cache_frame_data=False)

    plt.show()


# Example usage:
plot_gpu_memory_usage()
