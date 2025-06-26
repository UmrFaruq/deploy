import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# === 1. Ambil Dataset dari UCI ===
energy = fetch_ucirepo(id=242)
data = energy.data.original

# === 2. Tangani Missing Value (jika ada) ===
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# === 3. Pisahkan Fitur dan Target ===
X = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']]
y = data[['Y1', 'Y2']]  # Multi-output

# === 4. Normalisasi Fitur ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === 5. Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# === 6. Melatih Model Decision Tree Multi-Output ===
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === 7. Simpan Model dan Scaler ===
joblib.dump(model, 'model_energy_efficiency.pkl')
joblib.dump(scaler, 'scaler_energy.pkl')

print("✅ Model dan scaler berhasil disimpan!")
