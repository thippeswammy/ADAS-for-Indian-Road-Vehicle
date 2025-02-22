import os


def create_yolo_folder_structure(folder_name, base_path='', num_classes=None):
    """
    Create a YOLO folder structure with train, val, and test directories.
    Each directory contains images and labels subdirectories.
    If folders with the same name exist, the function appends a number to the folder name.
    It also creates a .yaml file with the dataset configuration.
    """
    if num_classes is None:
        num_classes = []
    if base_path == '':
        base_path = '..'
    while True:
        if folder_name == '':
            # print("folder_name is null")
            folder_name = input("Enter the folder name:")
            if folder_name == 'exit':
                exit(1)
        else:
            break

    def create_unique_folder(base, name):
        counter = 1
        new_name = name
        new_path = os.path.join(base, new_name)

        while os.path.exists(new_path):
            new_name = f"{name}_{counter}"
            new_path = os.path.join(base, new_name)
            counter += 1

        os.makedirs(new_path)
        return new_path

    # Create the main dataset folder with a unique name
    dataset_path = create_unique_folder(base_path, folder_name)

    # Create train, val, and test folders with unique names
    train_path = create_unique_folder(dataset_path, 'train')
    val_path = create_unique_folder(dataset_path, 'valid')
    test_path = create_unique_folder(dataset_path, 'test')

    # Create subdirectories for images and labels
    os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(test_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_path, 'labels'), exist_ok=True)

    # Create the .yaml file with the dataset configuration
    yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

# The number of classes in the dataset
nc: {len(num_classes)}

# Class names
names:
"""
    if len(num_classes) > 0:
        for i in range(len(num_classes)):
            yaml_content += f"  {i}: {num_classes[i]}\n"
    else:
        yaml_content += f"  0: class0\n"

    yaml_file_path = os.path.join(dataset_path, f"{folder_name}.yaml")
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write(yaml_content)

    # print(f"Folder structure created at: {dataset_path}")
    # print(f"YAML file created at: {yaml_file_path}")

    return folder_name, dataset_path


create_yolo_folder_structure()
