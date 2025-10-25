import torch
from torch import Tensor, nn, optim
from sklearn.datasets import fetch_california_housing, load_iris
from torch.utils.data import DataLoader, TensorDataset, random_split

# Multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_size):
        """Define layers here"""
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x : Tensor) -> Tensor:
        """Use layers here
        Input dimension: (batch_size, 8)
        Output dimension: (batch_size, 1)"""
        output = self.layers(x)
        return output

# Load the Boston housing dataset
data = fetch_california_housing()
inputs = data.data
targets = data.target

# Convert the data to tensors
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Normalize the data
mean = inputs.mean(dim=0, keepdim=True)
std = inputs.std(dim=0, keepdim=True)
inputs = (inputs - mean) / std

mean = targets.mean(dim=0, keepdim=True)
std = targets.std(dim=0, keepdim=True)
targets = (targets - mean) / std

print("input shape", inputs.shape)
print("target shape", targets.shape)
print(data.DESCR)

dataset = TensorDataset(inputs, targets)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
# Create the MLP model
model = MLP(8)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model
for epoch in range(100):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_dataloader:
        outputs = model(inputs)
        loss = criterion(outputs.flatten(), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate the model
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), targets)
            val_loss += loss.item()

        # Test the model
        test_loss = 0.0
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), targets)
            test_loss += loss.item()

    print(f"Epoch {epoch}, Train Loss: {train_loss / len(train_dataloader)}, Val Loss: {val_loss / len(val_dataloader)}, Test Loss: {test_loss / len(test_dataloader)}")
    
    if test_loss / len(test_dataloader) < 0.25:
        break