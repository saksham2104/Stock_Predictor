from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

class GradientBoostModel:
    def train(self, X_train, y_train):
        self.model = GradientBoostingRegressor()
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        return mean_absolute_error(y_test, y_pred)
