import os

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px

# ------------------ NLP / MODELO ------------------
import nltk
from nltk.stem import WordNetLemmatizer
import joblib

# Descargar wordnet (si ya está, no pasa nada)
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()


def split_into_lemmas(text):
    text = str(text).lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word, pos="v") for word in words]


# Ruta del modelo
MODELO_PATH = "modelo_cuoc_rf_compacto.pkl"

# Cargar el modelo una sola vez (Pipeline con CountVectorizer + RandomForest)
modelo = joblib.load(MODELO_PATH)

# ------------------ CATALOGO CUOC ------------------
# Archivo de catálogo oficial (ajusta el nombre si es distinto)
CUOC_FILE = "PerfilesOcupacionales-Excel-CUOC-2025.xlsx"

try:
    df_cuoc = pd.read_excel(CUOC_FILE)

    # Nos aseguramos de que el código CUOC sea string
    df_cuoc["CUOC"] = df_cuoc["CUOC"].astype(str).str.strip()

    # === IMPORTANTE ===
    # Cambia "Descripcion_ocupacion" por el nombre real de la columna de descripción
    DESC_COL = "Descripcion_ocupacion"
    # Si el nombre fuera distinto, por ejemplo "DESCRIPCION_OCUPACION", sería:
    # DESC_COL = "DESCRIPCION_OCUPACION"

    CUOC_DESCRIPCIONES = (
        df_cuoc.set_index("CUOC")[DESC_COL].fillna("").to_dict()
    )
except Exception as e:
    # Si hay error cargando el catálogo, dejamos el diccionario vacío
    print("Error cargando catálogo CUOC:", e)
    CUOC_DESCRIPCIONES = {}

# ------------------ APP DASH ------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Para despliegue (Railway, etc.)

app.layout = dbc.Container(
    [
        # ENCABEZADO
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src="/assets/logo_uniandes.png", style={"height": "70px"}
                    ),
                    width=2,
                ),
                dbc.Col(
                    html.H2(
                        "Herramienta de Recomendación de Códigos CUOC",
                        className="text-center mt-3",
                    ),
                    width=8,
                ),
                dbc.Col(
                    html.Img(
                        src="/assets/logo_spe.png", style={"height": "70px"}
                    ),
                    width=2,
                ),
            ],
            className="align-items-center my-3",
        ),
        html.Hr(),
        # SECCIÓN PRINCIPAL
        dbc.Row(
            [
                # Zona izquierda: entrada del perfil
                dbc.Col(
                    [
                        html.H5("Ingrese el perfil laboral del cargo solicitado:"),
                        dcc.Textarea(
                            id="perfil-texto",
                            placeholder=(
                                "Ejemplo: Se requiere ingeniero industrial con experiencia en "
                                "mejora de procesos, análisis de indicadores y gestión de calidad..."
                            ),
                            style={"width": "100%", "height": "250px"},
                        ),
                        html.Br(),
                        dbc.Button(
                            "Buscar códigos CUOC",
                            id="boton-buscar",
                            color="primary",
                            className="mt-2",
                        ),
                    ],
                    width=6,
                ),
                # Zona derecha: tabla de resultados
                dbc.Col(
                    [
                        html.H5("Códigos CUOC más relevantes:"),
                        html.Div(
                            id="resultados",
                            style={
                                "backgroundColor": "#f8f9fa",
                                "padding": "15px",
                                "borderRadius": "8px",
                                "minHeight": "250px",
                            },
                        ),
                    ],
                    width=6,
                ),
            ],
            className="my-4",
        ),
        html.Hr(),
        # METODOLOGÍA
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            "Notas de procesamiento", className="text-center mb-3"
                        ),
                        html.P(
                            "Esta herramienta utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) "
                            "para comparar el perfil ingresado por el prestador o empleador con las "
                            "descripciones ocupacionales oficiales de la Clasificación Única de "
                            "Ocupaciones para Colombia (CUOC). A partir de esta comparación semántica, "
                            "se presentan los cinco códigos más alineados con las competencias, "
                            "funciones y requisitos del cargo descrito.",
                            className="text-justify",
                        ),
                    ]
                )
            ],
            className="bg-light p-3 rounded mb-4",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("resultados", "children"),
    Input("boton-buscar", "n_clicks"),
    State("perfil-texto", "value"),
)
def recomendar_cuoc(n_clicks, perfil_texto):
    # Antes de dar clic o si está vacío
    if not n_clicks:
        return "Ingrese un perfil para obtener recomendaciones."
    if not perfil_texto or not perfil_texto.strip():
        return "El texto del perfil está vacío. Por favor ingréselo para obtener recomendaciones."

    try:
        texto = [perfil_texto]

        # Vector de probabilidades para cada clase CUOC
        proba = modelo.predict_proba(texto)[0]
        clases = modelo.classes_

        # Top 5 clases
        top_k = 5 if len(clases) >= 5 else len(clases)
        top_idx = np.argsort(proba)[::-1][:top_k]
        top_cuoc = clases[top_idx]
        top_probs = proba[top_idx]

        filas = []
        for cod, p in zip(top_cuoc, top_probs):
            cod_str = str(cod)
            desc = CUOC_DESCRIPCIONES.get(
                cod_str,
                "Descripción no disponible en el catálogo cargado. Consulte el catálogo oficial CUOC.",
            )
            filas.append(
                html.Tr(
                    [
                        html.Td(cod_str),
                        html.Td(desc),
                        html.Td(f"{p:.4f}"),
                    ]
                )
            )

        header = html.Thead(
            html.Tr(
                [
                    html.Th("Código CUOC"),
                    html.Th("Descripción"),
                    html.Th("Probabilidad"),
                ]
            )
        )

        body = html.Tbody(filas)

        tabla = html.Table(
            [header, body],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "marginTop": "10px",
            },
        )

        return tabla

    except Exception as e:
        return html.Div(
            [
                html.P("Ocurrió un error al generar las recomendaciones."),
                html.Pre(
                    str(e),
                    style={"whiteSpace": "pre-wrap", "fontSize": "12px"},
                ),
            ]
        )


if __name__ == "__main__":
    # Railway pone el puerto en la variable PORT
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
