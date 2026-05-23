import torch
import torch.nn.functional as F
from datasets import load_dataset

# Load
ds = load_dataset("codesignal/wine-quality", split="red")
data = ds.to_pandas()
print(f"Shape: {data.shape}")

# Prepare tensors
X = torch.tensor(data.drop(columns=["quality"]).values, dtype=torch.float32)
y = torch.tensor(data["quality"].values, dtype=torch.float32)

# Normalize inputs
X = (X - X.mean(dim=0)) / X.std(dim=0)

# Train/test split
n = int(0.8 * len(X))
Xtr, Xte = X[:n], X[n:]
ytr, yte = y[:n], y[n:]
print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

# MLP: 11 -> 64 -> 32 -> 1
torch.manual_seed(42)
W1 = torch.randn(11, 64) * 0.1; b1 = torch.zeros(64)
W2 = torch.randn(64, 32) * 0.1; b2 = torch.zeros(32)
W3 = torch.randn(32, 1) * 0.1;  b3 = torch.zeros(1)

params = [W1, b1, W2, b2, W3, b3]
for p in params:
    p.requires_grad = True

# Train
losses = []
for epoch in range(1000):
    h1 = torch.tanh(Xtr @ W1 + b1)
    h2 = torch.tanh(h1 @ W2 + b2)
    pred = (h2 @ W3 + b3).squeeze()
    loss = F.mse_loss(pred, ytr)

    loss.backward()

    lr = 0.01 if epoch < 500 else 0.001
    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()

    losses.append(loss.item())
    if epoch % 200 == 0:
        print(f"epoch {epoch:4d} | loss {loss.item():.4f}")

# Evaluate
with torch.no_grad():
    h1 = torch.tanh(Xte @ W1 + b1)
    h2 = torch.tanh(h1 @ W2 + b2)
    pred = (h2 @ W3 + b3).squeeze()
    test_loss = F.mse_loss(pred, yte)
    within_1 = ((pred - yte).abs() <= 1.0).float().mean()

print(f"\nTest MSE: {test_loss.item():.4f}")
print(f"Within ±1 of actual: {within_1.item()*100:.1f}%")
