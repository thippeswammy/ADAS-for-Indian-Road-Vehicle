import subprocess


def get_gpu_memory():
    """Gets the current GPU memory usage, including rendering GPU."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=gpu_name,memory.used,memory.total', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, text=True)
    memory_info = result.stdout.strip().split('\n')
    gpu_usage_info = []
    for info in memory_info:
        gpu_name, used, total = info.split(', ')
        used, total = int(used), int(total)
        usage_percentage = 100 * (used / total)
        gpu_usage_info.append((gpu_name, used, total, usage_percentage))
    return gpu_usage_info


print(get_gpu_memory())
