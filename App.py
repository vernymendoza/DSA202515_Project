import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # ENCABEZADO
    dbc.Row([
        dbc.Col(
            html.Img(src="/assets/logo_uniandes.png", style={"height": "70px"}),
            width=2
        ),
        dbc.Col(
            html.H2(
                "Herramienta de Recomendación de Códigos CUOC",
                className="text-center mt-3"
            ),
            width=8
        ),
        dbc.Col(
            html.Img(src="/assets/logo_spe.png", style={"height": "70px"}),
            width=2
        ),
    ], className="align-items-center my-3"),

    html.Hr(),

    # SECCIÓN PRINCIPAL
    dbc.Row([
        # Zona izquierda: entrada del perfil
        dbc.Col([
            html.H5("Ingrese el perfil laboral del cargo solicitado:"),
            dcc.Textarea(
                id="perfil-texto",
                placeholder=(
                    "Ejemplo: Se requiere ingeniero industrial con experiencia en "
                    "mejora de procesos, análisis de indicadores y gestión de calidad..."
                ),
                style={"width": "100%", "height": "250px"}
            ),
            html.Br(),
            dbc.Button(
                "Buscar códigos CUOC",
                id="boton-buscar",
                color="primary",
                className="mt-2"
            )
        ], width=6),

        # Zona derecha: tabla de resultados
        dbc.Col([
            html.H5("Códigos CUOC más relevantes:"),
            html.Div(
                id="resultados",
                style={
                    "backgroundColor": "#f8f9fa",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "minHeight": "250px"
                }
            )
        ], width=6),
    ], className="my-4"),

    html.Hr(),

    # METODOLOGÍA
    dbc.Row([
        dbc.Col([
            html.H4(
                "Notas de procesamiento",
                className="text-center mb-3"
            ),
            html.P(
                "Esta herramienta utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) "
                "para comparar el perfil ingresado por el prestador o empleador con las "
                "descripciones ocupacionales oficiales de la Clasificación Única de "
                "Ocupaciones para Colombia (CUOC). A partir de esta comparación semántica, "
                "se presentan los cinco códigos más alineados con las competencias, "
                "funciones y requisitos del cargo descrito.",
                className="text-justify"
            )
        ])
    ], className="bg-light p-3 rounded mb-4")
], fluid=True)


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

    # Simulación de recomendaciones (modelo)
    ejemplos_cuoc = [
        ("2145", "Ingenieros industriales y de producción"),
        ("3115", "Técnicos en control de calidad de manufactura"),
        ("1323", "Gerentes de producción industrial"),
        ("2423", "Profesionales en gestión de talento humano"),
        ("1211", "Directores de departamentos administrativos"),
    ]

# Construir tabla HTML
    header = html.Thead(
        html.Tr([
            html.Th("Código CUOC"),
            html.Th("Descripción")
        ])
    )

    rows = []
    for codigo, desc in ejemplos_cuoc:
        rows.append(
            html.Tr([
                html.Td(codigo),
                html.Td(desc)
            ])
        )

    body = html.Tbody(rows)

    tabla = html.Table(
        [header, body],
        style={
            "width": "100%",
            "borderCollapse": "collapse"
        }
    )

    # Estilos sencillos a nivel de celdas usando inline styles
    # (Dash no soporta CSS directo en cada <td> aquí, pero el navegador aplica por tabla)
    return tabla


if __name__ == "__main__":
    app.run(debug=True)
