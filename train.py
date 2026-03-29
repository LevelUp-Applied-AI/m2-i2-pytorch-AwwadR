import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.

    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        # 5 input features → 32 hidden units
        self.layer1 = nn.Linear(5, 32) 
        # activation function
        self.relu   = nn.ReLU()
        # 32 hidden → 1 output (price prediction)  
        self.layer2 = nn.Linear(32, 1) 

    def forward(self, x):
        """Define the forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 5).

        Returns:
            torch.Tensor: Predictions of shape (N, 1).
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def train_test_split_numpy(x, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    test_count = int(len(x) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    x_train = x.iloc[train_idx]
    x_test = x.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return x_train, x_test, y_train, y_test

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    """Load data, train HousingModel, and save predictions."""
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df = pd.read_csv("data/housing.csv")

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # ── 3. Train/test split ───────────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2, seed=42)

    # ── 4. Standardize using Train states only ───────────────────────────────────────────────
    x_train_mean = x_train.mean()
    x_train_std = x_train.std()

    x_train_scaled = (x_train - x_train_mean) / x_train_std
    x_test_scaled = (x_test - x_train_mean) / x_train_std


    # ── 5. Convert to Tensors ─────────────────────────────────────────────────
    x_train_tensor = torch.tensor(x_train_scaled.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    print(f"x_train shape: {x_train_tensor.shape}")
    print(f"x_test shape: {x_test_tensor.shape}")
    print(f"y_train shape: {y_train_tensor.shape}")
    print(f"y_test shape: {y_test_tensor.shape}")


    # ── 6. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # ── 7. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    loss_history = []

    for epoch in range(num_epochs +1):
        model.train()

        predictions = model(x_train_tensor)
        loss = criterion(predictions, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())


    # Print every 10 epochs: f"Epoch {epoch:3d}: Loss = {loss.item():.4f}"
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ── 8. Evaluation ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        train_predictions_tensor = model(x_train_tensor)
        test_predictions_tensor = model(x_test_tensor)
    
    train_predictions = train_predictions_tensor.numpy().flatten()
    test_predictions = test_predictions_tensor.numpy().flatten()

    y_train_np = y_train_tensor.numpy().flatten()
    y_test_np = y_test_tensor.numpy().flatten()


    # ── 9. Metrics ───────────────────────────────────────────────────
    train_mae = mae(y_train_np, train_predictions)
    test_mae = mae(y_test_np, test_predictions)

    train_r2 = r2_score(y_train_np, train_predictions)
    test_r2 = r2_score(y_test_np, test_predictions)

    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")


    # ── 10. Save Predictions ───────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "actual": y_test_np,
        "predicted": test_predictions
    })
    
    results_df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")

    # ── Plot 1: actual vs predicted ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_np, test_predictions)
    min_val = min(y_test_np.min(), test_predictions.min())
    max_val = max(y_test_np.max(), test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices (Test set)")
    plt.tight_layout()
    plt.savefig("predictions_plot.png")
    plt.close()
    print("Saved predictions_plot.png")

    # ── Plot 2: Loss curve ───────────────────────────────────────────────────
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()
    print("Saved loss_curve.png")


if __name__ == "__main__":
    main()
