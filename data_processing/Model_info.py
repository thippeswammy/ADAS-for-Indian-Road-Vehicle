import pandas as pd
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(
    r'F:\\RunningProjects\\YOLO_Model\\Training\\runs\\segment\\RoadSegmentationForMyDataset9\\weights\\best.pt').cuda()


# Recursive function to extract details from the model
def summarize_model(model):
    layers_summary = []

    def recurse(module, from_idx=-1, layer_idx=0):
        nonlocal layers_summary
        for name, layer in module.named_children():
            params = sum(p.numel() for p in layer.parameters())
            layer_name = layer._get_name()
            args = list(layer.parameters()) if hasattr(layer, 'parameters') else "None"
            layers_summary.append({
                "Layer": layer_idx,
                "From": from_idx,
                "N": 1,  # Default to 1
                "Params": params,
                "Module": layer_name,
                "Arguments": str(args),
            })
            layer_idx += 1
            # Recurse into submodules
            recurse(layer, from_idx=layer_idx, layer_idx=layer_idx)

    recurse(model.model, from_idx=-1, layer_idx=0)

    # Convert to Pandas DataFrame for easier visualization
    summary_df = pd.DataFrame(layers_summary)
    return summary_df


# Generate the summary
summary = summarize_model(model)
print(summary)

# Save the summary to a CSV file
summary.to_csv("yolo_model_detailed_summary.csv", index=False)

# model.info()
# YOLOv8l-seg summary: 401 layers, 45,936,819 parameters, 0 gradients, 220.8 GFLOPs
