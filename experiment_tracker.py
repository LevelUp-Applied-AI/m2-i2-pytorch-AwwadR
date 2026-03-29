import json
import time
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class HousingModel(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot))


def prepare_data():
    torch.manual_seed(42)
    np.random.seed(42)

    df = pd.read_csv("data/housing.csv")

    feature_cols = [
        "area_sqm",
        "bedrooms",
        "floor",
        "age_years",
        "distance_to_center_km",
    ]

    X = df[feature_cols]
    y = df[["price_jod"]]

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.8 * len(X))
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    X_train_mean = X_train.mean()
    X_train_std = X_train.std()

    X_train_scaled = (X_train - X_train_mean) / X_train_std
    X_test_scaled = (X_test - X_train_mean) / X_train_std

    X_train_tensor = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def run_experiment(X_train, X_test, y_train, y_test, learning_rate, hidden_size, num_epochs):
    torch.manual_seed(42)

    model = HousingModel(hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    loss_history = []

    for epoch in range(num_epochs):
        model.train()

        predictions = model(X_train)
        train_loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        loss_history.append(train_loss.item())

    train_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        train_preds_tensor = model(X_train)
        test_preds_tensor = model(X_test)

        final_train_loss = criterion(train_preds_tensor, y_train).item()
        final_test_loss = criterion(test_preds_tensor, y_test).item()

    train_preds = train_preds_tensor.numpy().flatten()
    test_preds = test_preds_tensor.numpy().flatten()
    y_train_np = y_train.numpy().flatten()
    y_test_np = y_test.numpy().flatten()

    train_mae = mae(y_train_np, train_preds)
    test_mae = mae(y_test_np, test_preds)
    train_r2 = r2_score(y_train_np, train_preds)
    test_r2 = r2_score(y_test_np, test_preds)

    return {
        "config": {
            "learning_rate": learning_rate,
            "hidden_size": hidden_size,
            "num_epochs": num_epochs,
        },
        "metrics": {
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "training_time_seconds": train_time,
        },
        "loss_history": loss_history,
    }


def print_leaderboard(experiments):
    ranked = sorted(experiments, key=lambda x: x["metrics"]["test_mae"])[:10]

    print("\nTop 10 Configurations by Test MAE")
    print("Rank | LR      | Hidden | Epochs | Test MAE    | Test R2   | Time (s)")
    print("-----|---------|--------|--------|-------------|-----------|--------")

    for i, exp in enumerate(ranked, start=1):
        cfg = exp["config"]
        m = exp["metrics"]
        print(
            f"{i:>4} | "
            f"{cfg['learning_rate']:<7} | "
            f"{cfg['hidden_size']:<6} | "
            f"{cfg['num_epochs']:<6} | "
            f"{m['test_mae']:<11.4f} | "
            f"{m['test_r2']:<9.4f} | "
            f"{m['training_time_seconds']:<7.2f}"
        )


def save_summary_plot(experiments):
    plt.figure(figsize=(10, 6))

    hidden_sizes = sorted(set(exp["config"]["hidden_size"] for exp in experiments))

    for hidden_size in hidden_sizes:
        subset = [
            exp for exp in experiments
            if exp["config"]["hidden_size"] == hidden_size and exp["config"]["num_epochs"] == 100
        ]
        subset = sorted(subset, key=lambda x: x["config"]["learning_rate"])

        x_vals = [exp["config"]["learning_rate"] for exp in subset]
        y_vals = [exp["metrics"]["test_mae"] for exp in subset]

        if x_vals:
            plt.plot(x_vals, y_vals, marker="o", label=f"hidden={hidden_size}, epochs=100")

    plt.xlabel("Learning Rate")
    plt.ylabel("Test MAE")
    plt.title("Experiment Summary: Test MAE vs Learning Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("experiment_summary.png")
    plt.close()


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    learning_rates = [0.001, 0.005, 0.01, 0.03]
    hidden_sizes = [16, 32, 64]
    num_epochs_list = [50, 100, 200]

    all_configs = list(itertools.product(learning_rates, hidden_sizes, num_epochs_list))
    print(f"Running {len(all_configs)} experiments...")

    experiments = []

    for i, (lr, hidden, epochs) in enumerate(all_configs, start=1):
        print(f"[{i}/{len(all_configs)}] lr={lr}, hidden={hidden}, epochs={epochs}")

        result = run_experiment(
            X_train, X_test, y_train, y_test,
            learning_rate=lr,
            hidden_size=hidden,
            num_epochs=epochs,
        )
        experiments.append(result)

    with open("experiments.json", "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    print("\nSaved experiments.json")

    print_leaderboard(experiments)
    save_summary_plot(experiments)

    best = min(experiments, key=lambda x: x["metrics"]["test_mae"])
    print("\nBest configuration:")
    print(best["config"])
    print(best["metrics"])


if __name__ == "__main__":
    main()