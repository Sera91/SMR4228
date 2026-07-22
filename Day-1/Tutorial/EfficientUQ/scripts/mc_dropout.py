import torch
import torch.nn as nn
import torch.nn.functional as F
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
X_train = np.concatenate([np.linspace(-0.3, -0.7, 20), np.linspace(0.3, 0.7, 20)])
y_train = true_function(X_train) + np.random.normal(scale=0.05, size=X_train.shape)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)


class DropoutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 100), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(100, 100), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(100, 1))

    def forward(self, x):
        return self.net(x)


# Training
model = DropoutMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-12)

for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    loss = 0.5  * (preds - y_train_tensor).pow(2).sum()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f} ")

# Predict
num_predictive_samples = 50
predictions = np.array(
    [model(X_tensor).detach().numpy().flatten() for _ in range(num_predictive_samples)]
)
mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)

# Plot Ensemble Predictions
fig = plt.figure(figsize=(6, 4))
ax = plt.gca()
for i, pred in enumerate(predictions):
    ax.plot(X, pred, color="black", alpha=0.01)
ax.plot(X, mean_pred, color="red", linewidth=2, label="Ensemble Mean")
ax.fill_between(
    X, mean_pred - std_pred, mean_pred + std_pred, color="steelblue", alpha=0.3
)
ax.scatter(X_train, y_train, color="red", edgecolor="black", zorder=3)
ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-1.2, 1.2)
fig.tight_layout()
fig.legend(loc="upper left", facecolor="white", framealpha=1.0)
plt.show()
