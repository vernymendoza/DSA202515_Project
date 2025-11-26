import os
from collections import Counter

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import pandas as pd
import joblib

# ===== NUEVO: NLTK + split_into_lemmas (para que el modelo pueda deserializar) =====
import nltk
from nltk.stem import WordNetLemmatizer

# Descargar wordnet (si ya está, no pasa nada)
nltk.download("wordnet")

wordnet_lemmatizer = WordNetLemmatizer()


def split_into_lemmas(text):
    text = str(text).lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]


# ============================================================
# 1. MODELO Y CATÁLOGO CUOC
# ============================================================

# Ruta del modelo compacto (pipeline CountVectorizer + RandomForest)
MODELO_PATH = "modelo_cuoc_rf_compacto.pkl"
modelo = joblib.load(MODELO_PATH)
# ------------------ CARGA CATÁLOGO CUOC ------------------
# Leemos empezando en la fila 4 (índice 3), donde están los encabezados
df_cuoc = pd.read_excel(
    "PerfilesOcupacionales-Excel-CUOC-2025.xlsx",
    header=3,
)

# Nos quedamos solo con las primeras 4 columnas y las renombramos por posición
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

# ============================================================
# 2. UTILIDADES PARA TEXTO (gráfico de palabras)
# ============================================================

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
    """Tokenizador simple para el gráfico de palabras."""
    texto = texto.lower()
    for ch in ",.;:()[]¿?¡!\"'/%-_\n\t\r":
        texto = texto.replace(ch, " ")
    tokens = [
        t for t in texto.split()
        if len(t) > 3 and t not in STOPWORDS_ES
    ]
    return tokens


# ============================================================
# 3. APP DASH
# ============================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Para despliegue (Railway, etc.)

app.layout = dbc.Container(
    [
        # ---------------- ENCABEZADO ----------------
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

        # ---------------- SECCIÓN PRINCIPAL ----------------
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
                # Zona derecha: resultados (tabla + gráficos)
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

        # ---------------- NOTAS DE PROCESAMIENTO ----------------
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

# ============================================================
# 4. CALLBACK PRINCIPAL
# ============================================================


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

        # Probabilidades para cada clase CUOC del modelo
        proba = modelo.predict_proba(texto)[0]
        clases = modelo.classes_

        # Top K clases
        top_k = 5 if len(clases) >= 5 else len(clases)
        top_idx = np.argsort(proba)[::-1][:top_k]
        top_cuoc = clases[top_idx]
        top_probs = proba[top_idx]

        # ---------------- TABLA DE RESULTADOS ----------------
        filas = []
        for cod, p in zip(top_cuoc, top_probs):
            cod_str = str(cod)
            nombre = CUOC_NOMBRES.get(cod_str, "Nombre no disponible")
            desc = CUOC_DESCRIPCIONES.get(
                cod_str,
                "Descripción no disponible en esta versión. Consulte el catálogo oficial CUOC.",
            )
            filas.append(
                html.Tr(
                    [
                        html.Td(cod_str),
                        html.Td(nombre),
                        html.Td(desc),
                        html.Td(f"{p:.4f}"),
                    ]
                )
            )

        header = html.Thead(
            html.Tr(
                [
                    html.Th("Código CUOC"),
                    html.Th("Nombre de la ocupación"),
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
                "fontSize": "13px",
            },
        )

        # ---------------- GRÁFICO 1: TOP 5 OCUPACIONES ----------------
        data_top = []
        for cod, p in zip(top_cuoc, top_probs):
            cod_str = str(cod)
            nombre = CUOC_NOMBRES.get(cod_str, "Nombre no disponible")
            # Etiqueta corta para el eje (código + nombre recortado)
            nombre_corto = (nombre[:55] + "…") if len(nombre) > 55 else nombre
            etiqueta = f"{cod_str} – {nombre_corto}"
            data_top.append(
                {
                    "Codigo": cod_str,
                    "Ocupacion": nombre,
                    "Etiqueta": etiqueta,
                    "Probabilidad": p,
                }
            )

        df_top = pd.DataFrame(data_top)

        fig_top5 = px.bar(
            df_top,
            x="Probabilidad",
            y="Etiqueta",
            orientation="h",
            text="Probabilidad",
        )

        fig_top5.update_traces(
            texttemplate="%{text:.4f}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.4f}<extra></extra>",
        )

        fig_top5.update_layout(
            title={
                "text": "<b>Top 5 ocupaciones CUOC (probabilidad del modelo)</b>",
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="<b>Probabilidad</b>",
            yaxis_title="<b>Ocupación CUOC</b>",
            margin=dict(l=10, r=10, t=60, b=40),
            height=350,
        )
        fig_top5.update_yaxes(automargin=True)

        # ---------------- GRÁFICO 2: PALABRAS CLAVE ----------------
        # Texto combinado: perfil + descripciones de las ocupaciones recomendadas
        texto_combinado = perfil_texto + " " + " ".join(
            CUOC_DESCRIPCIONES.get(str(c), "") for c in top_cuoc
        )

        tokens = limpiar_texto_simple(texto_combinado)
        conteo = Counter(tokens).most_common(10)

        if conteo:
            palabras, frecuencias = zip(*conteo)
            df_pal = pd.DataFrame(
                {"Palabra": list(palabras), "Frecuencia": list(frecuencias)}
            )

            fig_palabras = px.bar(
                df_pal,
                x="Palabra",
                y="Frecuencia",
                text="Frecuencia",
            )
            fig_palabras.update_traces(
                textposition="outside",
                hovertemplate="Palabra: %{x}<br>Frecuencia: %{y}<extra></extra>",
            )
            fig_palabras.update_layout(
                title={
                    "text": "<b>Palabras clave (perfil + ocupaciones recomendadas)</b>",
                    "x": 0.5,
                    "xanchor": "center",
                },
                xaxis_title="<b>Palabra</b>",
                yaxis_title="<b>Frecuencia</b>",
                margin=dict(l=10, r=10, t=60, b=80),
                height=350,
            )
        else:
            # Si no hay tokens suficientes, gráfico vacío pero agradable
            fig_palabras = px.bar()
            fig_palabras.update_layout(
                title={
                    "text": "<b>Palabras clave (perfil + ocupaciones recomendadas)</b>",
                    "x": 0.5,
                    "xanchor": "center",
                },
                annotations=[
                    dict(
                        text="Sin palabras clave suficientes",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.5,
                        y=0.5,
                        font=dict(size=14),
                    )
                ],
                xaxis_visible=False,
                yaxis_visible=False,
                margin=dict(l=10, r=10, t=60, b=40),
                height=350,
            )

        # ---------------- COMPONEMOS RESULTADO ----------------
        graficos = dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(
                        figure=fig_top5,
                        config={"displayModeBar": False},
                    ),
                    width=6,
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=fig_palabras,
                        config={"displayModeBar": False},
                    ),
                    width=6,
                ),
            ],
            className="mt-4",
        )

        return html.Div([tabla, graficos])

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


# ============================================================
# 5. EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    # Railway pone el puerto en la variable PORT
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
