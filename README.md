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


## Thursday Stretch — Experiment Tracker

For the stretch assignment, I built `experiment_tracker.py` to test different hyperparameter settings on the housing price model.

### Search grid
I tested 3 hyperparameters:
- Learning rate: 0.001, 0.005, 0.01, 0.03
- Hidden size: 16, 32, 64
- Epochs: 50, 100, 200

This gave 36 total experiments.

### What the script does
- Uses the same 80/20 train-test split for every run
- Trains a model for each configuration
- Records final train loss, final test loss, train/test MAE, train/test R², and training time
- Saves all results to `experiments.json`
- Prints a top-10 leaderboard ranked by lowest test MAE
- Saves `experiment_summary.png`

### Result summary
- Best configuration: `learning_rate=0.03, hidden_size=64, num_epochs=200`
- Best test MAE: `33731.4453`
- Best test R²: `-8.4911`
- Final train loss: `1368447872.0`
- Final test loss: `1230188928.0`

### Analysis
From the experiments, the best results came from the larger hidden size and more epochs. In general, 200 epochs performed better than 50 or 100, and the learning rate `0.03` gave the strongest results in this search. The model still did not get below 10,000 MAE, so next I would try scaling the target variable, testing more hidden sizes, or trying a different model structure.