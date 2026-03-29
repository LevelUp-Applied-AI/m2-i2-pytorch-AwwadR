# PyTorch Housing Price Prediction

## What the model predicts
This model predicts apartment price in Jordanian Dinars (`price_jod`) using five input features:
- `area_sqm`
- `bedrooms`
- `floor`
- `age_years`
- `distance_to_center_km`

## Training configuration
- Model: Neural network with layers `Linear(5, 32) -> ReLU -> Linear(32, 1)`
- Epochs: 100
- Learning rate: 0.03
- Optimizer: Adam
- Loss function: MSELoss
- Split: 80% train, 20% test

## Training outcome
The training loss decreased over the 100 epochs.

- Epoch 0 loss: `1918779392.0000`
- Epoch 50 loss: `1905505664.0000`
- Epoch 100 loss: `1847230848.0000`

## Evaluation metrics
- Train MAE: `40791.2812`
- Test MAE: `42227.0352`
- Train R²: `-8.8412`
- Test R²: `-7.9714`

## Observation
The loss kept decreasing during training, so the model was learning, but the evaluation metrics are still weak, which means the predictions are not very accurate yet.

## Overfitting discussion
The model does not look strongly overfitted because the train and test metrics are relatively close to each other. The test results are only a bit worse than the train results, so the bigger issue seems to be underfitting or that the model is still too weak, not extreme overfitting.

## What I would try next
- Train for more epochs
- Try a bigger or deeper model