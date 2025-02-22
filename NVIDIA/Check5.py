import tensorflow as tf


def check_tensorflow_cuda():
    # Check if TensorFlow is built with CUDA (GPU) support
    print("Is TensorFlow built with CUDA:", tf.test.is_built_with_cuda())

    # List available devices
    devices = tf.config.list_physical_devices()
    print("Available devices:")
    for device in devices:
        print(device)

    # Check if TensorFlow can access the GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("TensorFlow can access the following GPU(s):")
        for gpu in gpus:
            print(gpu)
    else:
        print("No GPU devices accessible to TensorFlow.")


def get_gpu_names():
    # List all physical devices available
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


if __name__ == "__main__":
    check_tensorflow_cuda()
    print("-" * 50)
    get_gpu_names()
