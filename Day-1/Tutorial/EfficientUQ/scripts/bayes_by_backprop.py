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


class BBBLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        prior_pi=0.5,
        prior_sigma1=1.0,
        prior_sigma2=0.002,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Variational parameters: mean and rho (for sigma)
        self.mu_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.rho_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4)
        )
        self.mu_bias = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.rho_bias = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        # Prior params
        self.prior_pi = prior_pi
        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2

    def forward(self, input):
        # Sample weights and biases with reparameterization trick
        sigma_weight = torch.log(1 + torch.exp(self.rho_weight))
        epsilon_w = torch.randn_like(sigma_weight)
        weight = self.mu_weight + sigma_weight * epsilon_w

        sigma_bias = torch.log(1 + torch.exp(self.rho_bias))
        epsilon_b = torch.randn_like(sigma_bias)
        bias = self.mu_bias + sigma_bias * epsilon_b

        # Save log probabilities
        self.log_qw = self._log_gaussian(weight, self.mu_weight, sigma_weight).sum()
        self.log_qb = self._log_gaussian(bias, self.mu_bias, sigma_bias).sum()
        self.log_pw = self._log_mixture_gaussian(weight).sum()
        self.log_pb = self._log_mixture_gaussian(bias).sum()

        return F.linear(input, weight, bias)

    def _log_gaussian(self, x, mu, sigma):
        return -0.5 * torch.log(2 * torch.pi * sigma**2) - (x - mu) ** 2 / (
            2 * sigma**2
        )

    def _log_mixture_gaussian(self, x):
        comp1 = torch.distributions.Normal(0, self.prior_sigma1).log_prob(x)
        comp2 = torch.distributions.Normal(0, self.prior_sigma2).log_prob(x)
        max_comp = torch.max(comp1, comp2)
        return max_comp + torch.log(
            self.prior_pi * torch.exp(comp1 - max_comp)
            + (1 - self.prior_pi) * torch.exp(comp2 - max_comp)
        )


class BBBMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(BBBLinear(1, 100), nn.ReLU(),
                                 BBBLinear(100, 100), nn.ReLU(),
                                 BBBLinear(100, 1))

    def forward(self, x):
        return self.net(x)

    def kl_loss(self):
        # Sum KL contributions from all layers
        return sum((layer.log_qw - layer.log_pw) + (layer.log_qb - layer.log_pb)
                   for layer in self.net if isinstance(layer, BBBLinear))


# Training
model = BBBMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    expected_likelihood = - 0.5  * (preds - y_train_tensor).pow(2).sum()
    kl = model.kl_loss() / len(X_train_tensor)
    loss = - (expected_likelihood - kl) # negative elbo
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(
            f"Epoch {epoch} - Loss: {loss.item():.4f} - "
            f"Expected Likelihood {expected_likelihood.item():.4f} - "
            f"KL {kl.item():.4f}"
        )

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
    ax.plot(X, pred, color="black", alpha=0.1)
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
