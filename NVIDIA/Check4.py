import psutil
import pyopencl as cl


def get_system_memory_info():
    # Get the virtual memory details
    virtual_memory = psutil.virtual_memory()
    print(f"Total system memory: {virtual_memory.total / (1024 * 1024)} MB")
    print(f"Available memory: {virtual_memory.available / (1024 * 1024)} MB")
    print(f"Used memory: {virtual_memory.used / (1024 * 1024)} MB")
    print(f"Memory usage percentage: {virtual_memory.percent}%")
    print("-" * 50)
    # Get the swap memory details
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


if __name__ == "__main__":
    get_system_memory_info()
    print("-----")
    get_amd_gpu_memory_info()
