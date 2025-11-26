import os

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import plotly.express as px  # (por si luego quieres usarlo)

# ------------------ NLP / MODELO ------------------
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Descargar wordnet (si ya está, no pasa nada)
nltk.download("wordnet")

wordnet_lemmatizer = WordNetLemmatizer()


def split_into_lemmas(text):
    text = str(text).lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word, pos="v") for word in words]


# Ruta del modelo (pipeline CountVectorizer + RandomForest)
MODELO_PATH = "modelo_cuoc_rf_compacto.pkl"
modelo = joblib.load(MODELO_PATH)

# ------------------ CATÁLOGO CUOC ------------------

CUOC_FILE = "PerfilesOcupacionales-Excel-CUOC-2025.xlsx"

try:
    df_cuoc = pd.read_excel(CUOC_FILE)

    # La fila 0 tiene los nombres reales de las columnas
    df_cuoc.columns = df_cuoc.iloc[0]  # fila 0 -> nombres de columnas
    df_cuoc = df_cuoc[1:]  # quitar esa fila de los datos

    # Renombrar a algo más manejable
    df_cuoc = df_cuoc.rename(
        columns={
            "Código del Gran Grupo": "GRAN_GRUPO",
            "Código de la Ocupación": "CUOC",
            "Nombre de la Ocupación": "NOMBRE",
            "Descripción de la Ocupación": "DESCRIPCION",
        }
    )

    # Normalizar códigos y descripciones
    df_cuoc["CUOC"] = df_cuoc["CUOC"].astype(str).str.strip()
    df_cuoc["DESCRIPCION"] = df_cuoc["DESCRIPCION"].fillna("")

    CUOC_DESCRIPCIONES = df_cuoc.set_index("CUOC")["DESCRIPCION"].to_dict()

    print("Diccionario CUOC cargado. Registros:", len(CUOC_DESCRIPCIONES))
    print("Ejemplo claves:", list(CUOC_DESCRIPCIONES.keys())[:10])

except Exception as e:
    print("Error cargando catálogo CUOC:", e)
    CUOC_DESCRIPCIONES = {}


# ------------------ FUNCIÓN AUXILIAR ------------------


def obtener_recomendaciones_y_palabras(perfil_texto, top_k=5, top_words=10):
    """
    Ejecuta el modelo y devuelve:
      - top_cuoc: códigos CUOC recomendados
      - top_probs: probabilidades correspondientes
      - df_palabras: dataframe con palabras clave y su frecuencia
    """
    texto = [perfil_texto]

    # Probabilidades del modelo
    proba = modelo.predict_proba(texto)[0]
    clases = modelo.classes_

    top_k = min(top_k, len(clases))
    top_idx = np.argsort(proba)[::-1][:top_k]
    top_cuoc = clases[top_idx]
    top_probs = proba[top_idx]

    # Palabras clave del texto (solo para visualización)
    vec = CountVectorizer(analyzer=split_into_lemmas, max_features=top_words)
    matriz = vec.fit_transform(texto)
    frec = matriz.toarray()[0]
    vocab = vec.get_feature_names_out()

    df_palabras = (
        pd.DataFrame({"palabra": vocab, "frecuencia": frec})
        .sort_values("frecuencia", ascending=False)
        .reset_index(drop=True)
    )

    return top_cuoc, top_probs, df_palabras


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
                        src="/assets/logo_uniandes.png",
                        style={"height": "70px"},
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
                        src="/assets/logo_spe.png",
                        style={"height": "70px"},
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
                        html.H5(
                            "Ingrese el perfil laboral del cargo solicitado:"
                        ),
                        dcc.Textarea(
                            id="perfil-texto",
                            placeholder=(
                                "Ejemplo: Se requiere ingeniero industrial con "
                                "experiencia en mejora de procesos, análisis de "
                                "indicadores y gestión de calidad..."
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
                # Zona derecha: tabla de resultados + gráficos
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
                        html.Br(),
                        dcc.Graph(
                            id="grafico-prob", style={"height": "260px"}
                        ),
                        html.Br(),
                        dcc.Graph(
                            id="grafico-palabras", style={"height": "260px"}
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
                            "Notas de procesamiento",
                            className="text-center mb-3",
                        ),
                        html.P(
                            "Esta herramienta utiliza técnicas de Procesamiento "
                            "de Lenguaje Natural (NLP) para comparar el perfil "
                            "ingresado por el prestador o empleador con las "
                            "descripciones ocupacionales oficiales de la "
                            "Clasificación Única de Ocupaciones para Colombia "
                            "(CUOC). A partir de esta comparación semántica, se "
                            "presentan los cinco códigos más alineados con las "
                            "competencias, funciones y requisitos del cargo "
                            "descrito.",
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


# ------------------ CALLBACKS ------------------


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
        top_cuoc, top_probs, _ = obtener_recomendaciones_y_palabras(
            perfil_texto
        )

        filas = []
        for cod, p in zip(top_cuoc, top_probs):
            cod_str = str(cod).strip()
            # Si viene como '43110.0' → '43110'
            if cod_str.endswith(".0"):
                cod_str = cod_str[:-2]

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
                html.P(
                    "Ocurrió un error al generar las recomendaciones."
                ),
                html.Pre(
                    str(e),
                    style={
                        "whiteSpace": "pre-wrap",
                        "fontSize": "12px",
                    },
                ),
            ]
        )


@app.callback(
    [
        Output("grafico-prob", "figure"),
        Output("grafico-palabras", "figure"),
    ],
    Input("boton-buscar", "n_clicks"),
    State("perfil-texto", "value"),
)
def actualizar_graficos(n_clicks, perfil_texto):
    # Si no hay clic o no hay texto, devolvemos figuras vacías
    if not n_clicks or not perfil_texto or not perfil_texto.strip():
        return go.Figure(), go.Figure()

    try:
        top_cuoc, top_probs, df_palabras = (
            obtener_recomendaciones_y_palabras(perfil_texto)
        )

        # Gráfico 1: ranking de CUOC por probabilidad
        codigos_str = []
        for c in top_cuoc:
            s = str(c).strip()
            if s.endswith(".0"):
                s = s[:-2]
            codigos_str.append(s)

        fig_prob = go.Figure(
            data=[go.Bar(x=codigos_str, y=top_probs)]
        )
        fig_prob.update_layout(
            title="Probabilidad por código CUOC recomendado",
            xaxis_title="Código CUOC",
            yaxis_title="Probabilidad",
            yaxis=dict(range=[0, 1]),
        )

        # Gráfico 2: palabras clave del perfil
        fig_pal = go.Figure(
            data=[
                go.Bar(
                    x=df_palabras["palabra"],
                    y=df_palabras["frecuencia"],
                )
            ]
        )
        fig_pal.update_layout(
            title="Palabras clave detectadas en el perfil",
            xaxis_title="Palabra",
            yaxis_title="Frecuencia",
        )

        return fig_prob, fig_pal

    except Exception as e:
        fig_err = go.Figure()
        fig_err.update_layout(
            title=f"Error generando gráficos: {e}"
        )
        return fig_err, go.Figure()


# ------------------ MAIN ------------------

if __name__ == "__main__":
    # Railway pone el puerto en la variable PORT
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
