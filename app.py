# Importación de librerías necesarias
import streamlit as st
import pickle
import numpy as np
import os

# Verificar si el modelo existe
if not os.path.exists("iris_model.pkl"):
    st.error("Error: No se encontró el archivo `iris_model.pkl`.")
    st.stop()

# Cargar el modelo desde `iris_model.pkl`
try:
    with open("iris_model.pkl", "rb") as f:
        model_dict = pickle.load(f)
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Obtener el modelo, el escalador y las clases objetivo
model = model_dict["model"]
scaler = model_dict["scaler"]
target_classes = model_dict["target_classes"]

# Función para hacer la predicción
def predict_iris(features):
    """Recibe un arreglo con características y devuelve la clase predicha."""
    features_scaled = scaler.transform([features])  # Escalar los datos de entrada
    prediction = model.predict(features_scaled)[0]  # Obtener la predicción
    return target_classes[prediction]

# Configuración de la interfaz de Streamlit
st.title("Clasificador de Iris 🌸")
st.write("Introduce las características de la flor y obtén su clasificación.")

# Crear sliders para ingresar valores
bill_length = st.slider("Longitud del sépalo (mm)", 4.0, 8.0, 5.1)
bill_depth = st.slider("Ancho del sépalo (mm)", 2.0, 4.5, 3.5)
flipper_length = st.slider("Longitud del pétalo (mm)", 1.0, 7.0, 1.4)
body_mass = st.slider("Ancho del pétalo (mm)", 0.1, 2.5, 0.2)

# Botón para hacer la predicción
if st.button("Predecir"):
    if None in [bill_length, bill_depth, flipper_length, body_mass]:
        st.error("Error: Todos los valores deben estar completos.")
        st.stop()
    
    features = [bill_length, bill_depth, flipper_length, body_mass]
    species = predict_iris(features)
    st.success(f"🔍 La flor pertenece a la especie: **{species}**")