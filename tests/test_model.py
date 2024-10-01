import pytest
from mlpredictor import MLPredictor


def test_train_and_predict():
    model = MLPredictor()
    model.train()
    result = model.predict([5.1, 3.5, 1.4, 0.2])
    assert len(result) == 1


if __name__ == "__main__":
    pytest.main()
