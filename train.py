import pandas as pd
import numpy as np
import torch
import torch.nn as nn


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


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df = pd.read_csv("data/housing.csv")

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # ── 3. Standardize Features ───────────────────────────────────────────────
    X_mean = X.mean()
    X_std  = X.std()
    X_scaled = (X - X_mean) / X_std
    # Why: features have very different scales; standardization ensures
    #      gradient updates are balanced across all input dimensions.

    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    print(f"X shape: {X_tensor.shape}")
    print(f"Y shape: {y_tensor.shape}")

    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    model = HousingModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    for epoch in range(num_epochs +1):
    #     Forward pass:  
        predictions = model(X_tensor)
    #     Compute loss:  
        loss = criterion(predictions, y_tensor)
    #     Zero grads:    
        optimizer.zero_grad()
    #     Backward:      
        loss.backward()
    #     Update:        
        optimizer.step()
    #     Print every 10 epochs: f"Epoch {epoch:3d}: Loss = {loss.item():.4f}"
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.4f}")

    # ── 7. Save Predictions ───────────────────────────────────────────────────
    with torch.no_grad():
        predictions_tensor = model(X_tensor)
    
    predictions_np = predictions_tensor.numpy().flatten()
    actuals_np = y_tensor.numpy().flatten()

    results_df = pd.DataFrame({
        "actual": actuals_np,
        "predicted": predictions_np
    })
    
    results_df.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
