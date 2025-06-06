import tensorflow as tf
from sklearn.metrics import mean_absolute_error

class NeuralNetModel:
    def train(self, X_train, y_train):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mae')
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()

    def evaluate(self, y_test, y_pred):
        return mean_absolute_error(y_test, y_pred)
