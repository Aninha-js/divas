from sklearn.linear_model import LinearRegression
import numpy as np

class LR:
    def __init__(self):
        self._model = LinearRegression()   # <-- agora é _model

    def train(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)

    def get_score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self._model.score(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_intercept(self) -> float:
        return self._model.intercept_      # <-- correto no sklearn

    def get_coefficients(self) -> np.ndarray:
        return self._model.coef_           # <-- correto no sklearn


if __name__ == "__main__":
    # Example usage
    X = np.array([[1], [2], [3]])
    y = np.array([3, 5, 7])

    model = LR()
    model.train(X, y)

    print("Score:", model.get_score(X, y))
    print("Predictions:", model.predict(X))
    print("Intercept:", model.get_intercept())
    print("Coefficients:", model.get_coefficients())
