# app.py
# -*- coding: utf-8 -*-

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import re

# ============================================================
# 0. Cargar modelo y diccionario CUOC
# ============================================================

with open("modelo_cuoc_tfidf_unigram_logreg.pkl", "rb") as f:
    modelo_cuoc = pickle.load(f)

dicc_cuoc = pd.read_csv("diccionario_cuoc.csv", dtype={"CUOC": str})
dicc_cuoc["CUOC"] = dicc_cuoc["CUOC"].str.zfill(5)
map_cuoc = dicc_cuoc.set_index("CUOC")["Descripcion_CUOC"].to_dict()

# ============================================================
# 1. Funciones de limpieza (mismas del notebook)
# ============================================================

def limpiar_texto(texto):
    if pd.isnull(texto):
        return texto
    return (texto
            .replace('\t', '')
            .replace('\n', '')
            .replace('\r', '')
            .replace('Ã¡', 'á')
            .replace('Ã©', 'é')
            .replace('Ã­', 'í')
            .replace('Ã³', 'ó')
            .replace('Ãº', 'ú')
            .replace('Ã±', 'ñ')
            .replace('Ã ', 'Á')
            .replace('Ã‰', 'É')
            .replace('Â¿', '¿')
            .replace('Â¡', '¡')
            .replace('â€¢', '')
            .replace('Â·', '')
            .replace('Ã?', 'Á')
            .replace('Ã','Á')
            .replace('Ã‰','É')
            .replace('Ã','Í')
            .replace('Â¡Ãš','Ú')
            .replace('Ãš','Ú')
            .replace('Ã','Ñ')
            .replace('â€¢', '')
            .replace('Â·', '')
           )

def palabrasTexto(texto):
    replacements = {
        'Distribuci\\?n': 'Distribución',
        'distribuci\\?n': 'distribución',
        'Producci\\?n': 'Producción',
        'producci\\?n': 'producción',
        'PRODUCCI\\?N': 'PRODUCCIÓN',
        'Dise\\?ador': 'Diseñador',
        'Gr\\?fico': 'Gráfico',
        'Telef\\?nica': 'Telefónica',
        'T\\?cnico': 'Técnico',
        'Tècnico': 'Técnico',
        'Tecn\\?logo': 'Tecnólogo',
        'Tecnol\\?gico': 'Tecnológico',
        'N\\?mina': 'Nómina',
        'BOGOT\\?': 'BOGOTÁ',
        'Bogot\\?': 'Bogotá',
        'Ch\\?a': 'Chía',
        'Log\\?stico': 'Logístico',
        'LOG\\?STICO': 'LOGÍSTICO',
        'Biling\\?e': 'Bilingue',
        'Bilingüe': 'Bilingue',
        'ü': 'u',
        'Aerol\\?nea': 'Aerolínea',
        'Cafeter\\?a': 'Cafetería',
        'Ferreter\\?a': 'Ferretería',
        'Met\\?licos': 'Metálicos',
        'Construcci\\?n': 'Construcción',
        'FRE\\?DO': 'FREÍDO',
        'Gesti\\?n': 'Gestión',
        'Mec\\?nico': 'Mecánico',
        'configuraci\\?n': 'configuración',
        'Cr\\?dito': 'Crédito',
        'C\\?cuta': 'Cúcuta',
        'Fabricaci\\?n': 'Fabricación',
        'VINCULACI\\?N': 'VINCULACIÓN',
        'COMPENSACI\\?N': 'COMPENSACIÓN',
        'TECNOLOG\\?A': 'TECNOLOGÍA',
        'Tem\\?tico': 'Temático',
        'topograf\\?a': 'topografía',
        'ALMAC\\?N': 'ALMACÉN',
        'Almac\\?n': 'Almacén',
        'contrataci\\?n': 'contratación',
        'Contrataci\\?n': 'Contratación',
        'env\\?os': 'envíos',
        'Env\\?os': 'Envíos',
        'ingl\\?s': 'inglés',
        'Electr\\?nica': 'Electrónica',
        'electr\\?nica': 'electrónica',
        'refrigeraci\\?n': 'refrigeración',
        'Refrigeraci\\?n': 'Refrigeración',
        'panader\\?a': 'panadería',
        'Pasteler\\?a': 'Pastelería',
        'Post\\?late': 'Postúlate',
        'campa\\?a': 'campaña',
        'Campa\\?a': 'Campaña',
        'Espa\\?ol': 'Español',
        'met\\?lica': 'metálica',
        'mec\\?nica': 'mecánica',
        'ferreter\\?a': 'ferretería',
        'pl\\?stico': 'plástico',
        'inyecci\\?n': 'inyección',
        'Atenci\\?n': 'Atención',
        'Ibagu\\?': 'Ibagué',
        'dom\\?sticos': 'domésticos',
        'Furg\\?n': 'Furgón',
        'Collar\\?n': 'Collarín',
        'Direcci\\?n': 'Dirección',
        'corrosi\\?n': 'corrosión',
        'h\\?brido': 'híbrido',
        '&': ' and ',
        'electroest\\?tico': 'electroestático',
        'af\\?n': 'afín'
    }
    for old, new in replacements.items():
        texto = texto.replace(old, new)
    return texto

def lowerTexto(texto):
    if pd.isnull(texto):
        return texto
    return texto.lower()

def trimTexto(texto):
    if pd.isnull(texto):
        return texto
    return re.sub(r'\s+', ' ', texto).strip()

def eliminar_espacios_numeros(texto):
    if pd.isnull(texto):
        return texto
    return re.sub(r'([0-9]+) ([0-9]+)', r'\1\2', texto)

def quitar_numeros(texto):
    if pd.isnull(texto):
        return texto
    return re.sub(r'\d+', '', texto)

def quitar_espacios_inicio(texto):
    if pd.isnull(texto):
        return texto
    return re.sub(r'^ +', '', texto)

def quitar_puntos_otros(texto):
    if pd.isnull(texto):
        return texto
    return (texto
            .replace('.', '')
            .replace(',', '')
            .replace(';', '')
            .replace('_', '')
            .replace('-', '')
            .replace('/', '')
            .replace("\\", '')
            .replace(')', '')
            .replace('(', '')
            .replace(':', '')
            .replace("'", '')
            .replace('?', '')
            .replace('¿', '')
            .replace('•', '')
            .replace('=', '')
            .replace('>', '')
            .replace('<', '')
            .replace('&', '')
            .replace('%', '')
            .replace('$', '')
            .replace('#', '')
            .replace('!', '')
            .replace('¡', '')
            .replace('+', '')
            .replace('@', '')
            .replace('nan', '')
           )

# stop_words tal como lo definiste (abreviado aquí; en tu código original está completo)
stop_words = set([
    "algún","alguna","algunas","alguno","algunos","ambos","ante","antes","aquel",
    # ... (puedes copiar TODO tu listado completo desde el notebook)
    "y","ya"
])

def eliminar_stop_words(texto):
    if pd.isnull(texto):
        return texto
    words = texto.split()
    filtered_words = [w for w in words if w not in stop_words]
    return ' '.join(filtered_words)

def limpiar_completo(texto):
    if pd.isnull(texto):
        return ""
    texto = limpiar_texto(texto)
    texto = palabrasTexto(texto)
    texto = lowerTexto(texto)
    texto = quitar_puntos_otros(texto)
    texto = eliminar_stop_words(texto)
    texto = trimTexto(texto)
    texto = eliminar_espacios_numeros(texto)
    texto = quitar_numeros(texto)
    texto = quitar_espacios_inicio(texto)
    return texto

# ============================================================
# 2. App Dash + layout (tu diseño)
# ============================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

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

# ============================================================
# 3. Callback de recomendación con el modelo
# ============================================================

@app.callback(
    Output("resultados", "children"),
    Input("boton-buscar", "n_clicks"),
    State("perfil-texto", "value")
)
def recomendar_cuoc_callback(n_clicks, perfil_texto):
    if not n_clicks:
        return "Ingrese un perfil para obtener recomendaciones."
    if not perfil_texto or not perfil_texto.strip():
        return "El texto del perfil está vacío. Por favor ingréselo para obtener recomendaciones."

    texto_proc = limpiar_completo(perfil_texto)

    proba = modelo_cuoc.predict_proba([texto_proc])[0]
    classes = modelo_cuoc.classes_
    topk_idx = np.argsort(proba)[-5:][::-1]
    codigos = classes[topk_idx]
    scores = proba[topk_idx]

    header = html.Thead(
        html.Tr([
            html.Th("Código CUOC"),
            html.Th("Descripción"),
            html.Th("Score (probabilidad)")
        ])
    )

    rows = []
    for codigo, score in zip(codigos, scores):
        desc = map_cuoc.get(codigo, "Descripción no encontrada")
        rows.append(
            html.Tr([
                html.Td(codigo),
                html.Td(desc),
                html.Td(f"{score:.4f}")
            ])
        )

    body = html.Tbody(rows)

    tabla = html.Table(
        [header, body],
        style={"width": "100%", "borderCollapse": "collapse"}
    )

    return tabla


if __name__ == "__main__":
    app.run_server(debug=True)
