import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

# def get_model_parameters(model):
#     params = []
#     for param in model.parameters():
#         params.append(param.data.cpu().numpy().flatten())
#     return np.concatenate(params)

# def set_model_parameters(model, param_vector):
#     """Sets the model's parameters from a single flattened NumPy array."""
#     current_index = 0
#     for param in model.parameters():
#         param_length = param.numel()
#         new_values = param_vector[current_index:current_index + param_length]
#         new_values = new_values.reshape(param.shape)
#         param.data = torch.tensor(new_values, dtype=torch.float32)
#         current_index += param_length

# def uniform_crossover(parent1_params, parent2_params):
#     """Performs uniform crossover between two parent parameter vectors."""
    
#     mask = np.random.rand(len(parent1_params)) < 0.5
#     child1_params = np.where(mask, parent1_params, parent2_params)
#     child2_params = np.where(mask, parent2_params, parent1_params)
    
#     return child1_params, child2_params

# def mutate(params, mutation_rate=0.1, mutation_strength=0.1):
#     """Applies Gaussian mutation to the parameters."""
#     mutation_mask = np.random.rand(len(params)) < mutation_rate
#     gaussian_noise = np.random.randn(len(params)) * mutation_strength
#     params[mutation_mask] += gaussian_noise[mutation_mask]
#     return params

# def breed(parent1, parent2, mutation_rate=0.1, mutation_strength=0.1):
#     """
#     Breeds two parent models to produce two children models using genetic algorithms.
    
#     Args:
#         parent1 (MyModel): The first parent model.
#         parent2 (MyModel): The second parent model.
#         mutation_rate (float): Probability of each gene being mutated.
#         mutation_strength (float): Standard deviation of the Gaussian noise added during mutation.
        
#     Returns:
#         child1 (MyModel): The first child model.
#         child2 (MyModel): The second child model.
#     """
#     # Extract parameters from parents
#     parent1_params = get_model_parameters(parent1)
#     parent2_params = get_model_parameters(parent2)
    
#     # Crossover
#     child1_params, child2_params = uniform_crossover(parent1_params, parent2_params)
    
#     # Mutation
#     child1_params = mutate(child1_params, mutation_rate, mutation_strength)
#     child2_params = mutate(child2_params, mutation_rate, mutation_strength)
    
#     # Create child models
#     child1 = MyModel()
#     child2 = MyModel()
    
#     # Set the new parameters
#     set_model_parameters(child1, child1_params)
#     set_model_parameters(child2, child2_params)
    
#     return child1, child2

def standardize(data, mean=None, std=None):
    """Standardizes data to zero mean and unit variance."""
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    return (data - mean) / std, mean, std

# Changed Line: Define a custom PyTorch Module
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Changed Line: Fully connected from 64 -> 16
        self.fc1 = nn.Linear(56, 16)
        
        # Changed Line: Fully connected from 16 -> 1
        self.fc2 = nn.Linear(16, 1)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights using uniform distribution
        nn.init.uniform_(self.fc1.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.fc2.weight, a=-1.0, b=1.0)
        
        # Alternatively, initialize biases using normal distribution
        nn.init.normal_(self.fc1.bias, mean=0.0, std=2.0)
        nn.init.normal_(self.fc2.bias, mean=0.0, std=2.0)
    
    def forward(self, x):
        # Changed Line: Hidden layer with sigmoid
        x = torch.relu(self.fc1(x))
        # Changed Line: Output layer with ReLU
        x = torch.relu(self.fc2(x))
        return x


x = torch.load("/Users/arasvalizadeh/Desktop/check project/Artificial-Intelligence-Project/phase2/DataSet.pt", map_location=torch.device("cpu"))
input_data = np.array(x['input_data'])
output_data = np.array(x['output_data'])

input_data_std, input_mean, input_std = standardize(input_data)
output_data_std, output_mean, output_std = standardize(output_data)
input_tensor = torch.tensor(input_data_std, dtype=torch.float32)
output_tensor = torch.tensor(output_data_std, dtype=torch.float32).unsqueeze(1)  # Ensure output is (N, 1)

criterion = nn.MSELoss()

# Use Adam optimizer
model = MyModel()  # Initialize the model
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Now model exists# Adjust learning rate as needed

batch_size = 64  # Adjust batch size as needed
epochs = 100  # Adjust number of epochs

# Convert data to PyTorch tensors
input_tensor = torch.tensor(input_data_norm, dtype=torch.float32)
output_tensor = torch.tensor(output_data_norm, dtype=torch.float32).unsqueeze(1)  # Ensure correct shape (N, 1)

# Create a DataLoader for batching
dataset = torch.utils.data.TensorDataset(input_tensor, output_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()  # Reset gradients
        
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


model.eval()  # Set model to evaluation mode

with torch.no_grad():  # No need for gradients in inference
    test_sample = torch.tensor(input_data_norm[:5], dtype=torch.float32)  # Take first 5 samples
    predicted = model(test_sample)

    # Convert back to original scale if using Z normalize
    predicted_original = predicted.numpy() * output_std + output_mean

    print("Predicted values:", predicted_original)
    print("Real values:",output_data_std[:5] * output_std + output_mean)



# # Instantiate two parent models
# parent1 = MyModel()
# parent2 = MyModel()

# # Optionally, print initial parameters for verification
# print("Parent 1 fc1 weights:\n", parent1.fc1.weight)
# print("Parent 2 fc1 weights:\n", parent2.fc1.weight)

# # Breed children
# child1, child2 = breed(parent1, parent2, mutation_rate=0.1, mutation_strength=0.1)

# # Print child parameters to verify
# print("\nChild 1 fc1 weights after crossover and mutation:\n", child1.fc1.weight)
# print("\nChild 2 fc1 weights after crossover and mutation:\n", child2.fc1.weight)

# # Example forward pass with children
# sample_input = torch.ones((1, 64))
# output_child1 = child1(sample_input)
# output_child2 = child2(sample_input)

# print("\nChild 1 Output:", output_child1.item())
# print("Child 2 Output:", output_child2.item())


