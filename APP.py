import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Título de la aplicación
st.title("Aplicación de Predicción de Cáncer de Mama")

# Cargar el escalador y el modelo
try:
    # Asegúrate de que los archivos estén en la misma carpeta que APP.py o proporciona la ruta completa
    standard_scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_log_reg_model.pkl')
    st.success("Escalador y modelo cargados exitosamente.")
except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'scaler.pkl' y 'best_log_reg_model.pkl' estén en la ruta correcta.")
    st.stop()

# --- PASO 1: Define TODAS las columnas que tu modelo espera, en el orden correcto ---
# IMPORTANTE: REEMPLAZA ESTA LISTA con las columnas EXACTAS de tu set de entrenamiento original
ALL_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# --- PASO 2: Crear dinámicamente los campos de entrada para cada característica ---
st.header("Ingrese los valores de las variables:")

# Usamos un diccionario para almacenar las entradas del usuario
input_dict = {}

# Creamos columnas para una mejor disposición en la UI
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

# Itera sobre todas las características para crear un campo de entrada para cada una
for i, feature in enumerate(ALL_FEATURES):
    with columns[i % 3]: # Distribuye las entradas en 3 columnas
        # Usamos st.number_input para cada característica
        input_dict[feature] = st.number_input(
            label=feature, 
            key=feature, 
            value=0.0, 
            format="%.4f"
        )

# --- PASO 3: Construir el DataFrame completo a partir de las entradas ---
input_df = pd.DataFrame([input_dict])


# --- PASO 4: Realizar la predicción cuando el usuario haga clic en el botón ---
if st.button("Predecir"):
    try:
        # Escalar el DataFrame completo
        input_data_scaled = standard_scaler.transform(input_df)
        
        # Realizar la predicción
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)

        # Mostrar el resultado de la predicción
        st.header("Resultado de la predicción:")
        
        if prediction[0] == 0:
            st.success("La predicción es: BENIGNO (Clase 0)")
            st.write(f"Probabilidad de ser benigno: {prediction_proba[0][0]*100:.2f}%")
            st.write(f"Probabilidad de ser maligno: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.error("La predicción es: MALIGNO (Clase 1)")
            st.write(f"Probabilidad de ser benigno: {prediction_proba[0][0]*100:.2f}%")
            st.write(f"Probabilidad de ser maligno: {prediction_proba[0][1]*100:.2f}%")

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
