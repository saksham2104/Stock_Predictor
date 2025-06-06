import os
import importlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.data_loader import download_data
from utils.feature_engineering import create_features
from utils.metrics import compute_mae

# === Step 1: Download and preprocess data ===
df = download_data("RELIANCE.NS", start="2018-01-01", end="2024-12-31")
df = create_features(df)

X = df[["Lag_1", "MA_5", "Volatility_5"]]
y = df["Close"]

# === Step 2: Train-test split and normalization ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Step 3: Load and evaluate models ===
model_dir = os.path.join(os.path.dirname(__file__), "models")
results = []

for filename in os.listdir(model_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        model_name = filename[:-3]
        module = importlib.import_module(f"models.{model_name}")  # <-- change here

        class_name = "".join([part.capitalize() for part in model_name.split('_')]) + "Model"
        ModelClass = getattr(module, class_name)

        model = ModelClass()
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = compute_mae(y_test, y_pred)

        print(f"{class_name} -> MAE: {mae:.4f}")
        results.append((class_name, mae))

# === Step 4: Save results ===
results_df = pd.DataFrame(results, columns=["Model", "MAE"])
results_df.to_csv("results.csv", index=False)
print("\n✅ Results saved to results.csv")
