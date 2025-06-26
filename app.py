import streamlit as st
import numpy as np

# Pilih salah satu:
# ========== SOLUSI 2: Jika model disimpan dengan joblib tanpa MultiOutputClassifier ==========
import joblib
model = joblib.load('model_energy_efficiency.pkl')

# ========== SOLUSI 3: Jika model disimpan dengan cloudpickle karena memakai MultiOutputClassifier ==========
# import cloudpickle
# with open("model_energy_efficiency.pkl", "rb") as f:
#     model = cloudpickle.load(f)

# Load scaler
scaler = joblib.load('scaler_energy.pkl')

# Judul Aplikasi
st.title("🔋 Prediksi Kebutuhan Energi Bangunan")

st.markdown("Masukkan nilai-nilai fitur bangunan untuk memprediksi kebutuhan **Heating Load (Y1)** dan **Cooling Load (Y2)**:")

# Input fitur
X1 = st.slider("Relative Compactness (X1)", 0.5, 1.0, 0.75)
X2 = st.slider("Surface Area (X2)", 500.0, 900.0, 700.0)
X3 = st.slider("Wall Area (X3)", 200.0, 420.0, 300.0)
X4 = st.slider("Roof Area (X4)", 100.0, 300.0, 150.0)
X5 = st.slider("Overall Height (X5)", 3.5, 7.0, 5.25)
X6 = st.selectbox("Orientation (X6)", [2, 3, 4, 5])
X7 = st.selectbox("Glazing Area (X7)", [0.0, 0.1, 0.25, 0.4])
X8 = st.selectbox("Glazing Area Distribution (X8)", [0, 1, 2, 3, 4, 5])

# Tombol Prediksi
if st.button("🔍 Prediksi Kebutuhan Energi"):
    # Buat array input
    input_data = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])
    
    # Normalisasi input
    input_scaled = scaler.transform(input_data)

    # Prediksi model
    prediction = model.predict(input_scaled)

    try:
        heating, cooling = prediction[0]
        st.success(f"🌡️ Heating Load (Y1): **{heating:.2f}**")
        st.success(f"❄️ Cooling Load (Y2): **{cooling:.2f}**")
    except Exception as e:
        st.warning("Format output model tidak sesuai ekspektasi.")
        st.write("Output:", prediction)
        st.error(str(e))

st.markdown("---")
st.caption("Dibuat dengan Streamlit • Dataset: UCI Energy Efficiency")
