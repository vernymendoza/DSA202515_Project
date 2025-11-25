# Manual 2 – Dashboard de Recomendación CUOC (Dash)

# Dashboard de Recomendación de Códigos CUOC  

Interfaz web para sugerir códigos CUOC a partir de descripciones de ofertas laborales  

Proyecto MLOps – Grupo 28  

---

## 📋 Descripción

Este dashboard fue desarrollado con **Dash** (framework web basado en Flask y Plotly) y permite:

- Ingresar el texto de una oferta laboral.
- Enviar ese texto a un **modelo de clasificación CUOC** ya entrenado.
- Mostrar los **códigos CUOC más probables** junto con su descripción y probabilidad.

El modelo se carga directamente en la aplicación (`modelo_cuoc_rf_compacto.pkl`), por lo que el dashboard funciona como **frontend + backend de predicción** en un solo servicio.

---

## 🎨 Características principales

- **Interfaz intuitiva**:
  - Área de texto amplia para pegar la descripción del cargo.
  - Botón único para lanzar la recomendación de códigos CUOC.
- **Resultados claros**:
  - Tabla con el **top 5** de códigos CUOC sugeridos.
  - Nombre/descripción de cada ocupación (cuando está en el diccionario).
  - Probabilidad estimada del modelo (cuando se expone en la versión final).
- **Contexto visual**:
  - Logo de la **Universidad de los Andes**.
  - Logo del **Servicio Público de Empleo (SPE)**.
- **Notas de metodología**:
  - Sección explicativa sobre cómo se construyó la recomendación (NLP + CUOC).

---

## 🏗️ Arquitectura del dashboard

Estructura de archivos relacionada con la aplicación web:

    app_final.py                  # Aplicación Dash principal
    modelo_cuoc_rf_compacto.pkl   # Modelo de recomendación CUOC serializado
    assets/
        logo_uniandes.png         # Logo Uniandes para el encabezado
        logo_spe.png              # Logo SPE para el encabezado
    requirements.txt              # Dependencias de Python
    Procfile                      # Comando de arranque (para Railway)

> Nota: la carpeta `assets/` es reconocida automáticamente por Dash para cargar imágenes y estilos.

---

## 🧩 Componentes de `app_final.py`

### 1. Carga del modelo

Al inicio de la aplicación se define la ruta al modelo y se carga con `joblib`:

- Se define un `MODELO_PATH` apuntando a `modelo_cuoc_rf_compacto.pkl`.
- Se invoca `joblib.load(MODELO_PATH)` para dejar el modelo en memoria.
- El modelo es un **pipeline de scikit-learn** que incluye:
  - `CountVectorizer` con lemas.
  - `RandomForestClassifier` multiclase.

De esta forma, la app puede llamar a `modelo.predict_proba([texto])` cada vez que el usuario envía una descripción.

### 2. Layout del dashboard

Elementos más importantes del layout:

- **Encabezado**:
  - Columna izquierda: `logo_uniandes.png`.
  - Columna central: título “Herramienta de Recomendación de Códigos CUOC”.
  - Columna derecha: `logo_spe.png`.

- **Zona izquierda (entrada)**:
  - Título: “Ingrese el perfil laboral del cargo solicitado:”.
  - `dcc.Textarea` con `id="perfil-texto"` para que el usuario ingrese el texto.
  - Botón `dbc.Button` con `id="boton-buscar"` y etiqueta “Buscar códigos CUOC”.

- **Zona derecha (salida)**:
  - Título: “Códigos CUOC más relevantes:”.
  - `html.Div` con `id="resultados"` donde se muestra la tabla con las recomendaciones.

- **Notas de procesamiento**:
  - Bloque de texto que describe en lenguaje sencillo:
    - Uso de NLP.
    - Uso de la clasificación CUOC.
    - Idea de similitud semántica entre la descripción y las ocupaciones oficiales.

### 3. Callback de recomendación

El flujo interactivo está dado por un callback de Dash:

- `Input("boton-buscar", "n_clicks")`  
- `State("perfil-texto", "value")`  
- `Output("resultados", "children")`

Lógica general del callback:

1. Si el usuario aún no ha dado clic, se muestra un mensaje de ayuda.
2. Si el texto está vacío o solo contiene espacios, se muestra un mensaje de validación.
3. Si hay texto válido:
   - Se construye una lista con ese texto (ej. `[perfil_texto]`).
   - Se llama a `modelo.predict_proba(...)`.
   - Se obtienen las clases del modelo (`modelo.classes_`).
   - Se ordenan las probabilidades de mayor a menor.
   - Se toma el **top 5** de códigos.
   - Se busca la descripción del CUOC (si existe un diccionario cargado desde el Excel).
   - Se construye una tabla HTML (`html.Table`) con filas del tipo:
     - Código CUOC.
     - Descripción.
     - Probabilidad.

---

## 🎯 Funcionalidades para el usuario

1. **Ingresar la descripción del cargo**  
   El usuario puede copiar/pegar el texto de la oferta o escribirlo manualmente.  
   Ejemplo:

   > “Se requiere profesional en ingeniería de sistemas con experiencia en desarrollo de software, análisis de datos y manejo de bases de datos relacionales…”

2. **Solicitar recomendaciones**  
   Al hacer clic en el botón **“Buscar códigos CUOC”**:
   - El texto se envía al modelo.
   - Se calcula el top de códigos CUOC.

3. **Interpretar la tabla de resultados**  
   La tabla muestra, para cada código:

   - **Código CUOC** (por ejemplo, `2519`).  
   - **Descripción de la ocupación** (por ejemplo, “Profesionales en informática no clasificados previamente”).  
   - **Probabilidad** (cuando esté desplegada), que indica cuánta confianza asigna el modelo a cada sugerencia.

4. **Entender el alcance**  
   - La herramienta **no reemplaza** la validación humana.
   - Sirve como apoyo para acelerar la búsqueda de códigos CUOC coherentes con el perfil descrito.

---

## 🚀 Instalación y uso local del dashboard

> Para el despliegue en la nube (Railway) ver el Manual 3.  
> Aquí se describe el uso local en el computador del usuario.

### 1. Clonar el repositorio y crear entorno

```bash
git clone https://github.com/vernymendoza/DSA202515_Project.git
cd DSA202515_Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verificar que el modelo exista

Confirmar que el archivo `modelo_cuoc_rf_compacto.pkl` está en la raíz del proyecto.  
Si no está, se puede generar siguiendo el **Manual 1** (entrenamiento y serialización).

### 3. Ejecutar el dashboard localmente

```bash
python app_final.py
```
- Por defecto la aplicación escucha en `http://localhost:8050` (si no se usa la variable de entorno `PORT`).  
   - Abrir esa URL en el navegador y probar la herramienta.

---

### 🔧 Configuración técnica

Al final de `app_final.py` se incluye:

```python
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
```

Esto permite:

- **En local:** si `PORT` no está definido, se usa el puerto `8050`.
- **En la nube (Railway):** se usa el valor de `PORT` que inyecta la plataforma.

---

## 🧪 Pruebas de funcionamiento

### 1. Prueba básica de carga

- Ejecutar `python app_final.py`.
- Verificar que se muestren título y logos sin errores.

### 2. Prueba de texto vacío

- Hacer clic en **“Buscar códigos CUOC”** sin escribir nada.
- Confirmar que aparece mensaje de texto vacío o equivalente.

### 3. Prueba con textos reales

- Probar varias descripciones de ofertas de diferentes perfiles.
- Validar que siempre se obtengan códigos CUOC razonables.

### 4. Prueba de robustez

- Probar textos muy cortos (ej. `auxiliar de bodega`).
- Probar textos muy largos (perfiles detallados).
- Verificar que la app no se rompa.

---

## 🚨 Troubleshooting (dashboard)

### La página no carga o aparece error 500

- Revisar la consola donde se ejecutó `python app_final.py`.
- Verificar que `modelo_cuoc_rf_compacto.pkl` exista y tenga el nombre correcto.
- Confirmar que `pip install -r requirements.txt` se ejecutó sin errores.

### Error `FileNotFoundError` al cargar el modelo

- Revisar que la ruta en `MODELO_PATH` coincida con la ubicación real del `.pkl`.
- En producción, asegurarse de hacer `git add` y `git push` incluyendo el modelo.

### El botón no hace nada / la tabla no se actualiza

Verificar ids:

- Botón: `"boton-buscar"`
- Textarea: `"perfil-texto"`
- Div resultados: `"resultados"`

Además, verificar que el callback usa exactamente esos ids en:

- `Input("boton-buscar", "n_clicks")`
- `State("perfil-texto", "value")`
- `Output("resultados", "children")`

### Errores de *encoding*

- Asegurar codificación **UTF-8** en los archivos.
- Para caracteres especiales (acentos, eñes), se recomienda mantener todo el flujo en UTF-8.
