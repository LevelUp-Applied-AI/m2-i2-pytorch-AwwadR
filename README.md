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

## Training outcome
The model trained successfully and the loss decreased over time, which shows the model learned patterns from the dataset.

Final loss value: Epoch 100: Loss = Epoch 100: Loss = 1880070016.0000

## Observation
Loss decreased steadily across training, and using a slightly higher learning rate helped the model learn faster within 100 epochs.