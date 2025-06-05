import pandas as pd
from ultralytics import YOLO


# Recursive function to extract details from model layers
def extract_layers(layer, layers_summary, from_id=-1):
    global current_layer_id
    for child_name, child in layer.named_children():
        params = sum(p.numel() for p in child.parameters() if p.requires_grad)  # Trainable parameters
        layer_type = child._get_name()  # Layer type (e.g., Conv, C2f, etc.)
        arguments = []

        # Extract meaningful arguments dynamically
        if hasattr(child, "kernel_size"):
            arguments.append(f"kernel_size={child.kernel_size}")
        if hasattr(child, "stride"):
            arguments.append(f"stride={child.stride}")
        if hasattr(child, "padding"):
            arguments.append(f"padding={child.padding}")
        if hasattr(child, "dilation"):
            arguments.append(f"dilation={child.dilation}")
        if hasattr(child, "activation"):
            arguments.append(f"activation={getattr(child.activation, '__name__', str(child.activation))}")

        # Append the current layer's details
        layers_summary.append({
            "Layer": current_layer_id,
            "From": from_id,
            "N": 1,
            "Params": params,
            "Module": layer_type,
            "Arguments": ", ".join(arguments) if arguments else "[]",
        })

        # Increment layer ID and recursively explore children
        current_layer_id += 1
        extract_layers(child, layers_summary, from_id=current_layer_id - 1)


# Function to summarize YOLO model layers
def summarize_yolo_model(model):
    global current_layer_id
    current_layer_id = 0  # Initialize layer ID counter
    layers_summary = []

    # Start extracting layers from the model
    extract_layers(model.model, layers_summary)

    # Convert summary to a DataFrame for visualization
    return pd.DataFrame(layers_summary)


# Load the YOLO model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()

# Generate the model summary
summary = summarize_yolo_model(model)

# Print the summary
print(summary)

# Optionally save to a CSV file
summary.to_csv("yolo_detailed_model_summary.csv", index=False)


