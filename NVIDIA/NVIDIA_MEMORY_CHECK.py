import subprocess
import pyopencl as cl


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    memory_info = result.stdout.strip().split('\n')
    for i, info in enumerate(memory_info):
        used, total = map(int, info.split(', '))
        print(f"GPU {i}: {used} MiB / {total} MiB used")


def get_amd_gpu_memory_usage():
    platforms = cl.get_platforms()
    for platform in platforms:
        for device in platform.get_devices(device_type=cl.device_type.GPU):
            print(f"GPU Name: {device.name}")
            print(f"GPU Memory Total: {device.global_mem_size / 1024 / 1024} MB")
            print("-" * 50)


if __name__ == "__main__":
    get_gpu_memory()
    print("-" * 50)
    get_amd_gpu_memory_usage()
