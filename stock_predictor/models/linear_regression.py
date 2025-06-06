from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class LinearRegressionModel:
    def train(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        return mean_absolute_error(y_test, y_pred)
