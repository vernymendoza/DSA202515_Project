import os
from collections import Counter

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd

# ------------------ NLP / MODELO ------------------
import nltk
from nltk.stem import WordNetLemmatizer
import joblib

# Solo necesitamos wordnet para el lematizador
nltk.download("wordnet")

wordnet_lemmatizer = WordNetLemmatizer()


def split_into_lemmas(text):
    text = str(text).lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word, pos="v") for word in words]


# Modelo compacto (pipeline CountVectorizer + RandomForest)
MODELO_PATH = "modelo_cuoc_rf_compacto.pkl"
modelo = joblib.load(MODELO_PATH)

# ------------------ CARGA CATÁLOGO CUOC  ------------------
# Ajusta el header si en tu archivo es distinto; este funciona con el que me mostraste
# ------------------ CARGA CATÁLOGO CUOC  ------------------
# Leemos empezando en la fila 4 (índice 3), donde están los encabezados de datos
df_cuoc = pd.read_excel(
    "PerfilesOcupacionales-Excel-CUOC-2025.xlsx", header=3
)

# Nos quedamos solo con las primeras 4 columnas y las renombramos por posición,
# para no depender de los nombres originales exactos del archivo.
df_cuoc = df_cuoc.iloc[:, :4]
df_cuoc.columns = [
    "gran_grupo",
    "CUOC",
    "nombre_ocupacion",
    "descripcion_ocupacion",
]

# Limpieza básica
df_cuoc = df_cuoc[df_cuoc["CUOC"].notna()]
df_cuoc["CUOC"] = df_cuoc["CUOC"].astype(int).astype(str).str.zfill(5)

CUOC_DESCRIPCIONES = dict(
    zip(df_cuoc["CUOC"], df_cuoc["descripcion_ocupacion"])
)
CUOC_NOMBRES = dict(zip(df_cuoc["CUOC"], df_cuoc["nombre_ocupacion"]))


# ------------------ STOPWORDS BÁSICAS (para gráfico de palabras) ------------------
STOPWORDS_ES = {
    "de",
    "la",
    "el",
    "y",
    "en",
    "para",
    "los",
    "las",
    "del",
    "con",
    "por",
    "un",
    "una",
    "se",
    "que",
    "al",
    "su",
    "sus",
    "a",
    "o",
    "u",
    "como",
    "sobre",
    "entre",
    "más",
    "menos",
    "etc",
}


def limpiar_texto_simple(texto: str) -> list[str]:
    """Tokenizador muy simple para el gráfico de palabras."""
    texto = texto.lower()
    for ch in ",.;:()[]¿?¡!\"'/%-_\n\t\r":
        texto = texto.replace(ch, " ")
    tokens = [
        t for t in texto.split() if len(t) > 3 and t not in STOPWORDS_ES
    ]
    return tokens


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
        # SECCIÓN PRINCIPAL (texto + tabla)
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
        # FILA DE GRÁFICOS
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        id="grafico-probabilidades",
                        figure=go.Figure(),
                    ),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(id="grafico-palabras", figure=go.Figure()),
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
                            "Esta herramienta utiliza técnicas de Procesamiento de "
                            "Lenguaje Natural (NLP) para comparar el perfil ingresado "
                            "por el prestador o empleador con las descripciones "
                            "ocupacionales oficiales de la Clasificación Única de "
                            "Ocupaciones para Colombia (CUOC). A partir de esta "
                            "comparación semántica, se presentan los cinco códigos "
                            "más alineados con las competencias, funciones y "
                            "requisitos del cargo descrito.",
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


# ------------------ HELPERS PARA GRÁFICOS ------------------
def figura_vacia(titulo: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=titulo,
        xaxis_visible=False,
        yaxis_visible=False,
        annotations=[
            dict(
                text="Sin datos disponibles",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                font=dict(size=14),
            )
        ],
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def crear_figura_probabilidades(top_cuoc, top_probs):
    if len(top_cuoc) == 0:
        return figura_vacia("Top 5 códigos CUOC (probabilidad del modelo)")

    codigos = [str(c) for c in top_cuoc]
    nombres = [CUOC_NOMBRES.get(c, "") for c in codigos]

    df_plot = pd.DataFrame(
        {
            "Código CUOC": codigos,
            "Probabilidad": top_probs,
            "Ocupación": nombres,
        }
    )

    fig = px.bar(
        df_plot,
        x="Código CUOC",
        y="Probabilidad",
        hover_data=["Ocupación"],
        text="Probabilidad",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        title="Top 5 códigos CUOC (probabilidad del modelo)",
        yaxis=dict(range=[0, float(max(top_probs) * 1.2)]),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def crear_figura_palabras(perfil_texto: str, descripciones_top: list[str]):
    texto_total = (perfil_texto or "") + " " + " ".join(descripciones_top)
    tokens = limpiar_texto_simple(texto_total)

    if not tokens:
        return figura_vacia(
            "Palabras clave (perfil + ocupaciones recomendadas)"
        )

    conteo = Counter(tokens)
    palabras, frec = zip(*conteo.most_common(10))

    fig = px.bar(
        x=list(palabras),
        y=list(frec),
    )
    fig.update_layout(
        title="Palabras clave (perfil + ocupaciones recomendadas)",
        xaxis_title="Palabra",
        yaxis_title="Frecuencia",
        margin=dict(l=40, r=20, t=80, b=80),
    )
    return fig


# ------------------ CALLBACK PRINCIPAL ------------------
@app.callback(
    Output("resultados", "children"),
    Input("boton-buscar", "n_clicks"),
    State("perfil-texto", "value")
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

        # Nombres bonitos para el gráfico (usamos nombre de ocupación)
        top_labels = [
            CUOC_NOMBRES.get(str(cod), f"CUOC {cod}")
            for cod in top_cuoc
        ]

        # ---------------- TABLA DE RESULTADOS ----------------
        filas = []
        for cod, p in zip(top_cuoc, top_probs):
            cod_str = str(cod)
            desc = CUOC_DESCRIPCIONES.get(
                cod_str,
                "Descripción no disponible en esta versión. Consulte el catálogo oficial CUOC."
            )
            filas.append(
                html.Tr([
                    html.Td(cod_str),
                    html.Td(desc),
                    html.Td(f"{p:.4f}")
                ])
            )

        header = html.Thead(
            html.Tr([
                html.Th("Código CUOC"),
                html.Th("Descripción"),
                html.Th("Probabilidad")
            ])
        )

        body = html.Tbody(filas)

        tabla = html.Table(
            [header, body],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "marginTop": "10px"
            }
        )

        # ---------------- GRÁFICO TOP 5 ----------------
        fig_top5 = go.Figure(
            data=[
                go.Bar(
                    x=top_labels,
                    y=top_probs,
                    text=[f"{p:.4f}" for p in top_probs],
                    textposition="auto",
                    customdata=np.array(top_cuoc).reshape(-1, 1),
                    hovertemplate=(
                        "<b>CUOC %{customdata[0]}</b><br>"
                        "%{x}<br>"
                        "Probabilidad: %{y:.4f}<extra></extra>"
                    ),
                )
            ]
        )

        fig_top5.update_layout(
            title="<b>Top 5 ocupaciones CUOC (probabilidad del modelo)</b>",
            xaxis_title="<b>Ocupación CUOC</b>",
            yaxis_title="<b>Probabilidad</b>",
            xaxis=dict(tickangle=-20),
            margin=dict(l=40, r=20, t=60, b=80),
            height=350,
        )

        # ---------------- GRÁFICO PALABRAS CLAVE ----------------
        # Texto combinado: perfil + descripciones de las ocupaciones recomendadas
        textos_union = perfil_texto + " "
        for cod in top_cuoc:
            cod_str = str(cod)
            textos_union += " " + CUOC_DESCRIPCIONES.get(cod_str, "")

        tokens = [
            w for w in split_into_lemmas(textos_union)
            if len(w) > 3  # palabras de al menos 4 letras
        ]
        conteo = Counter(tokens).most_common(10)
        if conteo:
            palabras, frecuencias = zip(*conteo)
        else:
            palabras, frecuencias = [], []

        fig_palabras = go.Figure(
            data=[
                go.Bar(
                    x=list(palabras),
                    y=list(frecuencias),
                    text=list(frecuencias),
                    textposition="auto",
                    hovertemplate="Palabra: %{x}<br>Frecuencia: %{y}<extra></extra>",
                )
            ]
        )

        fig_palabras.update_layout(
            title="<b>Palabras clave (perfil + ocupaciones recomendadas)</b>",
            xaxis_title="<b>Palabra</b>",
            yaxis_title="<b>Frecuencia</b>",
            xaxis=dict(tickangle=-30),
            margin=dict(l=40, r=20, t=60, b=80),
            height=350,
        )

        # ---------------- COMPONEMOS RESULTADO ----------------
        graficos = dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=fig_top5,
                        config={"displayModeBar": False}
                    ),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=fig_palabras,
                        config={"displayModeBar": False}
                    ),
                    width=6
                ),
            ],
            className="mt-4"
        )

        return html.Div([tabla, graficos])

    except Exception as e:
        return html.Div([
            html.P("Ocurrió un error al generar las recomendaciones."),
            html.Pre(str(e), style={"whiteSpace": "pre-wrap", "fontSize": "12px"})
        ])


if __name__ == "__main__":
    # Railway pone el puerto en la variable PORT
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
