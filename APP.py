import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Título de la aplicación
st.title("Aplicación de Predicción de Cáncer de Mama")

# Cargar el escalador y el modelo
try:
    standard_scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_log_reg_model.pkl')
    st.success("Escalador y modelo cargados exitosamente.")
except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'scaler.pkl' y 'best_log_reg_model.pkl' estén en la ruta correcta.")
    st.stop()


# Crear campos de entrada para las variables predictoras
st.header("Ingrese los valores de las variables:")

perimeter_worst = st.number_input("perimeter_worst")
concave_points_worst = st.number_input("concave points_worst")
concave_points_mean = st.number_input("concave points_mean")
area_worst = st.number_input("area_worst")
concavity_worst = st.number_input("concavity_worst")
area_se = st.number_input("area_se")


# Crear un DataFrame con los datos de entrada
input_data = pd.DataFrame({
    'perimeter_worst': [perimeter_worst],
    'concave_points_worst': [concave_points_worst],
    'concave_points_mean': [concave_points_mean],
    'area_worst': [area_worst],
    'concavity_worst': [concavity_worst],
    'area_se': [area_se]
})

# Escalar los datos de entrada
try:
    input_data_scaled = standard_scaler.transform(input_data)
    input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=input_data.columns)
except Exception as e:
    st.error(f"Error al escalar los datos: {e}")
    st.stop()

# Realizar la predicción
if st.button("Predecir"):
    try:
        prediction = model.predict(input_data_scaled_df)

        # Mostrar el resultado de la predicción
        st.header("Resultado de la predicción:")
        if prediction[0] == 0:
            st.success("La predicción es: Benigno (0)")
        else:
            st.error("La predicción es: Maligno (1)")

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
