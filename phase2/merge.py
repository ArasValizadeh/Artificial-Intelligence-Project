import torch
import os
import numpy as np

folder_path = "/Users/arasvalizadeh/Desktop/check project/Artificial-Intelligence-Project/phase2/agent_4"
output_file = os.path.join(folder_path, "/Users/arasvalizadeh/Desktop/check project/Artificial-Intelligence-Project/phase2/merged_dataset2.pt")

all_input_data = []
all_output_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pt"):
        file_path = os.path.join(folder_path, filename)
        
        try:
            data = torch.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        if not all(k in data for k in ["input_data", "output_data"]):
            print(f"Skipping {file_path}: Missing keys")
            continue

        input_data, output_data = data["input_data"], data["output_data"]

        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)  # Convert to NumPy array first
            input_data = torch.tensor(input_data)  # Then convert to tensor

        if isinstance(output_data, list):
            output_data = np.array(output_data, dtype=np.float32)  # Convert to NumPy array first
            output_data = torch.tensor(output_data) 

        print(f"File: {filename}, Input Shape: {input_data.shape}, Output Shape: {output_data.shape}")

        all_input_data.append(input_data)
        all_output_data.append(output_data)

if all_input_data and all_output_data:
    try:
        print(len(all_input_data))
        merged_input = torch.cat(all_input_data, dim=0)
        merged_output = torch.cat(all_output_data, dim=0)
        
        torch.save({"input_data": merged_input, "output_data": merged_output}, output_file)
        print(f"Merged dataset saved to {output_file}")
    except Exception as e:
        print(f"Error during merging: {e}")
else:
    print("No valid data found to merge.")