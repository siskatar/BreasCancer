import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io

# Título de la aplicación
st.title("Aplicación de Predicción de Cáncer de Mama")
st.write("Esta aplicación predice si un tumor es benigno o maligno basado en un archivo de datos.")

# Cargar el escalador y el modelo
try:
    # Asegúrate de que los archivos estén en la misma carpeta que APP.py o proporciona la ruta completa
    standard_scaler = joblib.load('scaler.pkl')
    model = joblib.load('best_log_reg_model.pkl')
except FileNotFoundError:
    st.error("Error: Asegúrate de que los archivos 'scaler.pkl' y 'best_log_reg_model.pkl' estén en la ruta correcta.")
    st.stop()

# --- Listas de características ---
# Lista completa para el escalador (DEBE COINCIDIR CON TU ENTRENAMIENTO)
ALL_FEATURES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]
# Las 6 características específicas para el modelo de Regresión Logística
SELECTED_FEATURES = [
    'perimeter_worst', 'concave points_worst', 'concave points_mean', 
    'area_worst', 'concavity_worst', 'area_se'
]

st.divider()

# --- PASO 1: Cargar el archivo de datos ---
st.header("Paso 1: Cargar Archivo de Datos")

st.info(
    "Por favor, suba un archivo Excel (`.xlsx`, `.xls`) o CSV que contenga el conjunto de datos completo (30 columnas). "
    "Puede descargar una plantilla para asegurarse de que el formato sea el correcto."
)

# Crear un DataFrame de plantilla para descargar
template_df = pd.DataFrame(columns=ALL_FEATURES)
template_csv = template_df.to_csv(index=False).encode('utf-8')

st.download_button(
   label="Descargar plantilla completa (CSV)",
   data=template_csv,
   file_name='plantilla_prediccion_completa.csv',
   mime='text/csv',
)

uploaded_file = st.file_uploader("Elija un archivo para analizar", type=['xlsx', 'xls', 'csv'], label_visibility="collapsed")


# --- PASO 2 y 3: Procesar, predecir y descargar ---
if uploaded_file is not None:
    try:
        # Cargar los datos del archivo
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)
        
        # --- Validación de Columnas ---
        missing_cols = set(ALL_FEATURES) - set(input_df.columns)
        if missing_cols:
            st.error(f"Error: Faltan las siguientes columnas en el archivo: {', '.join(missing_cols)}")
            st.stop()
        
        # Asegurar el orden correcto de las 30 columnas para el escalador
        input_df_full_ordered = input_df[ALL_FEATURES]

        # Escalar el DataFrame completo
        input_data_scaled_array = standard_scaler.transform(input_df_full_ordered)
        
        # Convertir el array escalado de vuelta a un DataFrame para seleccionar las columnas
        scaled_df_full = pd.DataFrame(input_data_scaled_array, columns=ALL_FEATURES)
        
        # Seleccionar solo las 6 características necesarias para el modelo
        model_input_df = scaled_df_full[SELECTED_FEATURES]

        # Realizar la predicción
        prediction = model.predict(model_input_df)
        prediction_proba = model.predict_proba(model_input_df)

        # Usamos el DataFrame original para mostrar los resultados
        results_df = input_df.copy()
        results_df['Predicción'] = ['MALIGNO' if p == 1 else 'BENIGNO' for p in prediction]
        results_df['Confianza'] = [f"{p.max()*100:.2f}%" for p in prediction_proba]
        
        st.divider()

        # --- PASO 2: Mostrar Resultados ---
        st.header("Paso 2: Visualizar Predicciones")
        
        # Aplicar estilo para resaltar la predicción
        def highlight_prediction(row):
            color = 'lightcoral' if row['Predicción'] == 'MALIGNO' else 'lightgreen'
            # Aplica el color a toda la fila para mayor visibilidad
            return [f'background-color: {color}'] * len(row)

        # Crear el objeto Styler para mostrar Y para descargar
        styled_df = results_df.style.apply(highlight_prediction, axis=1)

        st.dataframe(styled_df)
        
        st.divider()

        # --- PASO 3: Descargar Resultados ---
        st.header("Paso 3: Descargar Archivo con Predicciones")
        
        # Crear un buffer en memoria para guardar el archivo Excel
        output = io.BytesIO()
        # Escribir el DataFrame estilizado en el buffer de Excel
        styled_df.to_excel(output, engine='openpyxl', index=False)
        excel_data = output.getvalue()
        
        st.download_button(
           label="Descargar resultados (Excel)",
           data=excel_data,
           file_name='resultados_prediccion_cancer.xlsx',
           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")

