import GPUtil


def gpu_memory_usage():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.memoryTotal} MB")
        print(f"GPU Memory Used: {gpu.memoryUsed} MB")
        print(f"GPU Memory Free: {gpu.memoryFree} MB")
        print(f"GPU Memory Utilization: {gpu.memoryUtil * 100}%")


gpu_memory_usage()
