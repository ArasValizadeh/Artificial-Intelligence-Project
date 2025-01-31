import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


x = torch.load("/Users/arasvalizadeh/Desktop/check project/Artificial-Intelligence-Project/phase2/Dataset/merged_dataset.pt", 
               map_location=torch.device("cpu"))

input_data = np.asarray(x['input_data'], dtype=np.float32)  
output_data = np.asarray(x['output_data'], dtype=np.float32) 


scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)  
input_mean, input_std = scaler.mean_, scaler.scale_  

input_tensor = torch.tensor(input_data, dtype=torch.float32)
output_tensor = torch.tensor(output_data, dtype=torch.float32).unsqueeze(1)  


class MyModel(nn.Module):
    def __init__(self, input_mean, input_std):
        super(MyModel, self).__init__()
        self.input_mean = torch.tensor(input_mean, dtype=torch.float32)  # Store mean
        self.input_std = torch.tensor(input_std, dtype=torch.float32)  # Store std

        self.fc1 = nn.Linear(56, 256)  
        self.fc2 = nn.Linear(256, 64)  
        self.fc3 = nn.Linear(64, 8)   
        self.fc4 = nn.Linear(8, 1)    
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize all layers with Normal distribution"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            # nn.init.normal_(layer.weight, mean=0.0, std=2.0)
            # nn.init.normal_(layer.bias, mean=2.0, std=0.5)
            nn.init.xavier_uniform_(layer.weight)  # Xavier initialization
            nn.init.zeros_(layer.bias)  # Initialize biases to zero

    def forward(self, x):
        """Forward pass through network with automatic normalization"""
        x = (x - self.input_mean) / self.input_std

        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x) 
        return x


model = MyModel(input_mean, input_std)

criterion = nn.L1Loss()  # Huber Loss
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=9e-2) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.4)  # Reduce LR every 500 epochs

batch_size = 512
epochs = 2000


dataset = torch.utils.data.TensorDataset(input_tensor, output_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


for epoch in range(epochs):
    epoch_loss = 0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)  
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step() 

    epoch_loss /= len(dataloader)  
    print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {epoch_loss:.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'input_mean': input_mean,
    'input_std': input_std
}, "best_model_with_norm96.pth")



model.eval()
with torch.no_grad():
    test_sample = torch.tensor(input_data[:30], dtype=torch.float32)  
    predicted = model(test_sample)  

    print("\nPredicted values:", predicted)
    print("Real values:", output_data[:30])