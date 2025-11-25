# DSA202515_Project
Repositorio del proyecto del trabajo final para la materia de Despliegues de Soluciones  Analíticas. 

*Grupo 28*
Integrado por:
- *Tatiana Cardenas*
- *Verny Mendoza*
- *David Castiblanco*
- *Holman Zarta*

#### *Nota: Este es el repositorio desarrollado por el equipo, enfocado en la clasificacion única de ocupaciones en Colombia.*

# Manual 1 – Modelo CUOC (Entrenamiento y Serialización)

# Clasificador de Códigos CUOC para Ofertas de Empleo  
Herramienta de recomendación de códigos CUOC a partir de descripciones textuales

Proyecto MLOps – Grupo 28

---

## 📋 Descripción

Este repositorio contiene el flujo de datos y el modelo de Machine Learning para **recomendar códigos CUOC** (Clasificación Única de Ocupaciones para Colombia) a partir del **texto de la oferta de empleo**.

Dado el texto de una oferta laboral, el modelo sugiere los códigos CUOC más probables asociados a ese perfil.

---

## 🎯 Objetivo

Predecir automáticamente el **código CUOC más probable** (y candidatos alternativos) a partir de:

- La descripción de la oferta de empleo (`Descripcion_oferta`).
- El código CUOC asignado históricamente (`CUOC`), usado como etiqueta de entrenamiento.

La herramienta devuelve:

- Un conjunto de **códigos CUOC recomendados**.
- Un **top 5** de códigos con sus probabilidades estimadas.

---

## 🏗️ Arquitectura del proyecto

Estructura principal del repositorio:

```text
.
├── assets/                              # Logos usados en el dashboard
│   ├── logo_uniandes.png
│   └── logo_spe.png
├── Ofertas_proyecto_U_DSA202515.parquet.dvc   # Dataset de ofertas versionado con DVC
├── PerfilesOcupacionales-Excel-CUOC-2025.xlsx # Catálogo oficial CUOC
├── Proyecto_DSA_grupo28.ipynb           # Notebook de exploración y entrenamiento
├── app_final.py                         # Aplicación web Dash (modelo embebido)
├── modelo_cuoc_rf_compacto.pkl          # Modelo entrenado y compactado (RandomForest)
├── requirements.txt                     # Dependencias del proyecto
├── Procfile                             # Start command para Railway
└── README.md                            # Documentación general del proyecto
```
- La estructura puede incluir archivos adicionales (.dvc, imágenes, etc.), pero los anteriores son los componentes centrales del modelo y su despliegue.

---

## 📊 Datasets utilizados

### 1. Ofertas laborales históricas

**Archivo:** `Ofertas_proyecto_U_DSA202515.parquet`  
(seguido mediante `Ofertas_proyecto_U_DSA202515.parquet.dvc`)

Variables principales:

- `Descripcion_oferta`: texto completo de la oferta de empleo.  
- `CUOC`: código CUOC etiquetado para esa oferta.  
- Otras variables (ciudad, nivel educativo, etc.), usadas principalmente para análisis exploratorio.

Este dataset es la base para entrenar el clasificador CUOC.

### 2. Catálogo CUOC oficial

**Archivo:** `PerfilesOcupacionales-Excel-CUOC-2025.xlsx`

Variables principales:

- `CUOC`: código de ocupación.  
- `Descripción`: nombre de la ocupación.  
- Información adicional sobre funciones y perfiles ocupacionales.

En el modelo se usa principalmente para:

- Construir un diccionario de códigos → descripciones.  
- Enriquecer la presentación de resultados en el dashboard.

---

## 🔧 Preprocesamiento y balanceo

### Limpieza y normalización de texto

- Conversión a minúsculas.  
- Tokenización simple por espacios.  
- **Lematización** con NLTK (`WordNetLemmatizer`) para reducir palabras a su forma base (énfasis en verbos).

Función utilizada (a nivel conceptual):

- Recibe el texto.
- Lo pasa a minúsculas.
- Separa en palabras por espacios.
- Aplica lematización verbo a verbo.

Esta función se pasa como `analyzer` al `CountVectorizer`, de modo que el mismo preprocesamiento se aplique tanto en entrenamiento como en predicción.

### Balanceo de clases: `adaptive_resample`

El problema original presenta fuerte desbalance entre códigos CUOC (muchas clases con pocos registros).  
Se define una función de remuestreo que:

- Para clases con pocos registros: realiza **sobremuestreo con reemplazo** hasta un número objetivo (`target_samples`).  
- Para clases muy frecuentes: realiza **submuestreo suave**, limitando el tamaño máximo de registros por clase.

Resultado: un dataset de entrenamiento **más balanceado**, que mejora la capacidad del modelo para sugerir códigos menos frecuentes.

---

## 🤖 Modelo entrenado

### Tipo de modelo

- **Vectorización:** `CountVectorizer` con `analyzer=split_into_lemmas`.  
- **Clasificador:** `RandomForestClassifier` multiclase.

Para hacer el modelo viable en entornos con memoria limitada (como Railway), se entrenó una **versión compacta**:

- `max_features` en `CountVectorizer` reducido (vocabulario limitado).  
- Árboles con profundidad acotada (`max_depth`).  
- Número de árboles moderado (`n_estimators`).  
- Límite en el tamaño del conjunto de entrenamiento.

Ejemplo de pipeline (esquemático):

- Etapa 1: `CountVectorizer` con lemas, máximo ~3000 términos y `min_df` para filtrar términos muy raros.  
- Etapa 2: `RandomForestClassifier` con alrededor de 40 árboles, profundidad máxima 15, `max_features="sqrt"` y `min_samples_leaf=5`.

### Entrenamiento y evaluación

Flujo general:

1. Carga del dataset de ofertas.  
2. Aplicación de `adaptive_resample` para balancear clases.  
3. Separación en conjunto de entrenamiento y prueba (`train_test_split`).  
4. Entrenamiento del pipeline `pipe_cuoc_rf`.  
5. Evaluación con métricas de clasificación (accuracy y análisis por clase).

Métrica global (aprox.):

- **Accuracy:** ~0.39  

Dado que es un problema multiclase con muchas clases y fuerte desbalance, la métrica se interpreta como una **línea base razonable** para demostrar el flujo completo de MLOps y despliegue.

### Serialización del modelo

Una vez entrenado, el modelo se guarda como archivo `.pkl`:

- Se utiliza `joblib.dump` para serializar el pipeline completo.
- El archivo resultante es `modelo_cuoc_rf_compacto.pkl`, que contiene:
  - El vectorizador entrenado.  
  - El clasificador Random Forest entrenado.  

Este archivo es el que se carga dentro de la aplicación Dash para inferencia en producción.

---

## 🚀 Instalación y uso local

### 1. Clonar el repositorio

    git clone https://github.com/vernymendoza/DSA202515_Project.git
    cd DSA202515_Project

### 2. Crear entorno virtual e instalar dependencias

    python -m venv .venv

Activar entorno:

- Windows:  
  `.venv\Scripts\activate`
- Linux/Mac:  
  `source .venv/bin/activate`

Instalar dependencias:

    pip install --upgrade pip
    pip install -r requirements.txt

### 3. (Opcional) Recuperar datos con DVC

Si se tiene configurado el remoto DVC:

    dvc pull

Esto descarga el dataset de ofertas y otros artefactos versionados.

### 4. Reentrenar el modelo (opcional)

Abrir el notebook:

    jupyter notebook Proyecto_DSA_grupo28.ipynb

Seguir las secciones de:

- Carga de datos.  
- Preprocesamiento y balanceo.  
- Entrenamiento.  
- Evaluación.  
- Exportación del modelo como `modelo_cuoc_rf_compacto.pkl`.

Si no se desea reentrenar, se puede utilizar directamente el `.pkl` incluido en el repositorio.

---

## 🧩 Artefactos generados

- `modelo_cuoc_rf_compacto.pkl`  
  Modelo de clasificación CUOC listo para uso en producción (consumido por la app Dash).

- `Proyecto_DSA_grupo28.ipynb`  
  Notebook con:
  - Análisis exploratorio.  
  - Preprocesamiento y balanceo.  
  - Entrenamiento y evaluación.  
  - Generación del modelo serializado.

- Archivos `.dvc`  
  Referencias a los datasets gestionados con DVC.

---

## 🚨 Troubleshooting

- **Error al cargar el modelo (`FileNotFoundError`)**  
  Verificar que `modelo_cuoc_rf_compacto.pkl` exista en la raíz del proyecto y que el código apunte a ese nombre (`MODELO_PATH = "modelo_cuoc_rf_compacto.pkl"`).

- **Incompatibilidades de `scikit-learn`**  
  Asegurar que la versión instalada sea compatible con el modelo entrenado.  
  Se recomienda instalar siempre desde `requirements.txt`.

- **Problemas de memoria al reentrenar**  
  Reducir:
  - `max_features` del `CountVectorizer`.  
  - `n_estimators` y `max_depth` del `RandomForestClassifier`.  
  - Tamaño de la muestra de entrenamiento.

---

## 👥 Equipo – Grupo 28

- *Tatiana Cardenas*
- *Verny Mendoza*
- *David Castiblanco*
- *Holman Zarta*

Este proyecto hace parte del curso **Despliegue de Soluciones Analíticas (MLOps) – MIAD, Universidad de los Andes**.






