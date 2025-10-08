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

# --- PASO 1: Define las 6 columnas EXACTAS que tu modelo espera ---
# Estas son las características con las que tu modelo fue entrenado
SELECTED_FEATURES = [
    'perimeter_worst', 'concave points_worst', 'concave points_mean', 
    'area_worst', 'concavity_worst', 'area_se'
]

# --- PASO 2: Crear un cargador de archivos para el Excel ---
st.header("Cargar archivo para predicción")

st.info(
    "Por favor, suba un archivo Excel (`.xlsx`, `.xls`) o CSV que contenga las 6 columnas necesarias para el modelo. "
    "El orden no importa, pero los nombres deben coincidir."
)

# Crear un DataFrame de plantilla para descargar
template_df = pd.DataFrame(columns=SELECTED_FEATURES)
template_csv = template_df.to_csv(index=False).encode('utf-8')

st.download_button(
   label="Descargar plantilla de 6 columnas (CSV)",
   data=template_csv,
   file_name='plantilla_prediccion_6_features.csv',
   mime='text/csv',
)


uploaded_file = st.file_uploader("Elija un archivo", type=['xlsx', 'xls', 'csv'])

# --- PASO 3: Procesar el archivo y realizar la predicción ---
if uploaded_file is not None:
    try:
        # Cargar los datos del archivo
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
        
        st.subheader("Datos cargados del archivo:")
        st.dataframe(input_df)

        # --- Validación de Columnas ---
        missing_cols = set(SELECTED_FEATURES) - set(input_df.columns)
        if missing_cols:
            st.error(f"Error: Faltan las siguientes columnas en el archivo: {', '.join(missing_cols)}")
            st.stop()
        
        # Asegurar el orden correcto de las 6 columnas seleccionadas
        input_df_ordered = input_df[SELECTED_FEATURES]

        # Escalar el DataFrame completo (que ahora solo tiene 6 columnas)
        input_data_scaled = standard_scaler.transform(input_df_ordered)
        
        # Realizar la predicción
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)

        # --- Mostrar Resultados ---
        st.header("Resultados de la predicción:")
        
        # Crear un DataFrame con los resultados
        # Usamos el DataFrame ordenado para mantener solo las 6 columnas
        results_df = input_df_ordered.copy()
        results_df['Predicción'] = ['MALIGNO' if p == 1 else 'BENIGNO' for p in prediction]
        results_df['Confianza'] = [f"{p.max()*100:.2f}%" for p in prediction_proba]
        
        # Aplicar estilo para resaltar la predicción
        def highlight_prediction(row):
            color = 'lightcoral' if row['Predicción'] == 'MALIGNO' else 'lightgreen'
            # Creamos una lista de estilos vacía del tamaño de la fila
            styles = [''] * len(row)
            # Aplicamos el color solo a la columna 'Predicción'
            styles[row.index.get_loc('Predicción')] = f'background-color: {color}'
            return styles

        st.dataframe(results_df.style.apply(highlight_prediction, axis=1))


    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
