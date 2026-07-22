import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Create synthetic data
def true_function(x):
    return np.sin(3 * x)

X = np.linspace(-1.5, 1.5, 100)
y_true = true_function(X)
X_train = np.concatenate([
    np.linspace(-0.3, -0.7, 20),
    np.linspace(0.3, 0.7, 20)
])
y_train = true_function(X_train) + np.random.normal(scale=0.05, size=X_train.shape)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

# Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 100), nn.ReLU(), 
                                 nn.Linear(100, 100), nn.ReLU(),
                                 nn.Linear(100, 1))
    def forward(self, x):
        return self.net(x)

# Train ensemble
def train_model(data):
    X_train_tensor, y_train_tensor = data
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.4f} ")
    return model

# Train ensemble of models
n_ensemble = 15
data = (X_train_tensor, y_train_tensor)
models=[train_model(data) for _ in range(n_ensemble)]


# Predict
predictions = np.array([model(X_tensor).detach().numpy().flatten() for model in models])
mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)

# Plot Ensemble Predictions
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
for i, pred in enumerate(predictions):
    ax.plot(X, pred, color='black', alpha=0.1)
ax.plot(X, mean_pred , color='red', linewidth=2, label='Ensemble Mean')
ax.fill_between(X, mean_pred - std_pred, mean_pred + std_pred, color='steelblue', alpha=0.3)
ax.scatter(X_train, y_train, color='red', edgecolor='black', zorder=3)
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-1.2, 1.2)
fig.tight_layout()
fig.legend(loc='upper left', facecolor='white', framealpha=1.0)
plt.show()

