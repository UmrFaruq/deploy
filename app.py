import streamlit as st
import numpy as np
import joblib

from sklearn.multioutput import MultiOutputClassifier  # ⬅ Tambahkan baris ini
from sklearn.preprocessing import MinMaxScaler

# Load model dan scaler
model = joblib.load('model_energy_efficiency.pkl')
scaler = joblib.load('scaler_energy.pkl')

st.title("Prediksi Kebutuhan Energi Bangunan 🏢")

st.markdown("Masukkan nilai-nilai fitur bangunan di bawah ini:")

# Input fitur
X1 = st.slider("Relative Compactness (X1)", 0.5, 1.0, 0.75)
X2 = st.slider("Surface Area (X2)", 500.0, 900.0, 700.0)
X3 = st.slider("Wall Area (X3)", 200.0, 420.0, 300.0)
X4 = st.slider("Roof Area (X4)", 100.0, 300.0, 150.0)
X5 = st.slider("Overall Height (X5)", 3.5, 7.0, 5.25)
X6 = st.selectbox("Orientation (X6)", [2, 3, 4, 5])
X7 = st.selectbox("Glazing Area (X7)", [0.0, 0.1, 0.25, 0.4])
X8 = st.selectbox("Glazing Area Distribution (X8)", [0, 1, 2, 3, 4, 5])

# Tombol prediksi
if st.button("Prediksi Kebutuhan Energi"):
    # Bentuk array input
    user_input = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])

    # Normalisasi fitur
    user_input_scaled = scaler.transform(user_input)

    # Prediksi menggunakan model
    prediction = model.predict(user_input_scaled)

    # Jika multi-output
    try:
        heating, cooling = prediction[0]
        st.success(f"🔸 **Heating Load (Y1):** {heating:.2f}")
        st.success(f"🔸 **Cooling Load (Y2):** {cooling:.2f}")
    except:
        st.warning("Prediksi tidak dalam format multi-output.")
        st.write("Hasil:", prediction)

st.markdown("---")
st.caption("Dibuat dengan Streamlit | Data: UCI Energy Efficiency Dataset")
