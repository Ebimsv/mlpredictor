# MLPredictor

MLPredictor is a simple machine learning package that trains a RandomForest model using the Iris dataset and enables users to make predictions. The package is built using `scikit-learn` and is intended as a demonstration of packaging Python machine learning projects for distribution.

## Features

- Train a RandomForestClassifier on the Iris dataset.
- Make predictions on new data after training.
- Save and load trained models.

## Installation

You can install the package via **PyPI** or from **source**.

### Install from PyPI

```bash
pip install mlpredictor
```

### Install from Source (GitHub)

```bash
git clone https://github.com/ebimsv/mlpredictor.git
cd mlpredictor
pip install .
```

## Usage

After installation, you can use `MLPredictor` to train a model and make predictions.

### Example: Training and Making Predictions

```python
from mlpredictor import MLPredictor

# Initialize the predictor
predictor = MLPredictor()

# Train the model on the Iris dataset
predictor.train()

# Make a prediction on a sample input
sample_input = [5.1, 3.5, 1.4, 0.2]
prediction = predictor.predict(sample_input)

print(f"Predicted class: {prediction}")
```

### Example: Saving and Loading a Model

You can also save the trained model to a file and load it later for future predictions:

```python
from mlpredictor import MLPredictor

# Initialize and train the model
predictor = MLPredictor()
predictor.train()

# Save the trained model to a file
predictor.save_model("my_model.pkl")

# Load the model from the file and make predictions
new_predictor = MLPredictor()
new_predictor.load_model("my_model.pkl")

sample_input = [5.1, 3.5, 1.4, 0.2]
prediction = new_predictor.predict(sample_input)

print(f"Predicted class: {prediction.item()}")
```

## Testing

This project includes a test suite. You can run tests using `pytest`:

```bash
pip install pytest
pytest tests
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues if you have any suggestions or find bugs.

### Key Updates:

1. **Installation Instructions**: Added how to install from both PyPI and source (GitHub).
2. **Usage Instructions**: Provided examples for training, making predictions, saving the model, and loading a saved model.
3. **Testing Section**: Added instructions on how to run the tests using `pytest`.
4. **Contributing Section**: Encouraged contributions for further improvements.

This updated README should provide users with all the information they need to install, use, and contribute to your project.
