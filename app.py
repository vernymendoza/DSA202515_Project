import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re

st.set_page_config(page_title="Recomendador de ocupaciones CUOC", layout="wide")


# -------------------------------------------------------------------
# 1. Cargar modelo y diccionario CUOC
# -------------------------------------------------------------------

@st.cache_resource
def load_model():
    with open("modelo_cuoc_tfidf_unigram_logreg.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_diccionario():
    # diccionario_cuoc.csv debe tener columnas: CUOC, Descripcion_CUOC
    df = pd.read_csv("diccionario_cuoc.csv", dtype={"CUOC": str})
    df["CUOC"] = df["CUOC"].str.zfill(5)
    mapping = df.set_index("CUOC")["Descripcion_CUOC"].to_dict()
    return mapping

modelo = load_model()
map_cuoc = load_diccionario()

# -------------------------------------------------------------------
# 2. Funciones de preprocesamiento
#    (PEGA AQUÍ la misma función limpiar_completo del notebook)
# -------------------------------------------------------------------

def limpiar_texto(texto):
    # EJEMPLO MUY SIMPLE; REEMPLÁZALO POR TU limpiar_completo
    if texto is None:
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9 áéíóúñ]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# Si prefieres, pega aquí toda tu función 'limpiar_completo' y usa esa en vez de 'limpiar_texto'.


# -------------------------------------------------------------------
# 3. Función de recomendación Top-5
# -------------------------------------------------------------------

def recomendar_cuoc(texto, modelo, k=5):
    texto_proc = limpiar_texto(texto)  # o limpiar_completo(texto)

    proba = modelo.predict_proba([texto_proc])[0]
    classes = modelo.classes_

    topk_idx = np.argsort(proba)[-k:][::-1]
    codigos = classes[topk_idx]
    scores = proba[topk_idx]

    resultados = []
    for c, s in zip(codigos, scores):
        nombre = map_cuoc.get(c, "Descripción no encontrada")
        resultados.append(
            {
                "Código CUOC": c,
                "Nombre de la ocupación": nombre,
                "Score (probabilidad)": round(float(s), 4),
            }
        )
    return pd.DataFrame(resultados)

# -------------------------------------------------------------------
# 4. Interfaz de Streamlit
# -------------------------------------------------------------------


st.title("Recomendador de ocupaciones CUOC")
st.write(
    """
    Ingrese la descripción de la oferta de empleo y la herramienta recomendará 
    los **5 códigos CUOC** más afines, a partir del histórico de ofertas del Servicio Público de Empleo.
    """
)

with st.form("form_cuoc"):
    texto_usuario = st.text_area(
        "Descripción de la oferta de empleo",
        height=200,
        placeholder="Ejemplo: Se requiere profesional en ingeniería de sistemas con experiencia en desarrollo de software..."
    )
    submitted = st.form_submit_button("Recomendar códigos CUOC")

if submitted:
    if not texto_usuario.strip():
        st.warning("Por favor ingrese una descripción para generar recomendaciones.")
    else:
        st.subheader("Top 5 ocupaciones recomendadas")
        df_res = recomendar_cuoc(texto_usuario, modelo, k=5)
        st.dataframe(df_res, use_container_width=True)

        st.caption(
            "Los scores corresponden a la probabilidad estimada por el modelo de que la oferta pertenezca a cada ocupación CUOC."
        )

# (Opcional) Pequeña sección informativa
with st.expander("Acerca del modelo"):
    st.write(
        """
        El modelo está basado en una representación TF-IDF de la descripción de la oferta y 
        una Regresión Logística multiclase. El desempeño obtenido en la muestra de prueba fue 
        de aproximadamente **26% de accuracy Top-1** y **61% de accuracy Top-5**.
        """
    )