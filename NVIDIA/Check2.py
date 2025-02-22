import pyopencl as cl
import subprocess

platforms = cl.get_platforms()
for platform in platforms:
    print("Platform:", platform.name)
    for device in platform.get_devices():
        print("  Device:", device.name)


def check_amd_gpu():
    try:
        output = subprocess.check_output(['clinfo'], universal_newlines=True)
        return 'AMD' in output
    except subprocess.CalledProcessError:
        return False


if check_amd_gpu():
    print("AMD GPU is present.")
else:
    print("No AMD GPU found.")
