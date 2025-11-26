# Manual 3 – Despliegue del Dashboard CUOC en Railway  
# Servicio web para recomendación de códigos CUOC en la nube  

Proyecto MLOps – Grupo 28  

---

## 📋 Descripción

Este manual describe de forma detallada el proceso para desplegar en producción el dashboard de recomendación de códigos CUOC (archivo app_final.py) utilizando la plataforma Railway.
El objetivo es dejar documentado el procedimiento completo para que cualquiera pueda:

•	Replicar el despliegue directamente desde el repositorio en GitHub.
•	Comprender cómo Railway construye la imagen, instala dependencias y ejecuta el servicio.
•	Saber dónde consultar logs y cómo diagnosticar errores durante el build y la ejecución.

---

## 🧱 Requisitos previos

Antes de desplegar en Railway se debe contar con:

- Repositorio en GitHub con al menos estos archivos en la rama `main`:

  - `app_final.py`
  - `modelo_cuoc_rf_compacto.pkl`
  - `requirements.txt`
  - `Procfile`
  - Carpeta `assets/` con los logos (`logo_uniandes.png`, `logo_spe.png`)

- Cuenta en [Railway](https://railway.app/) vinculada a GitHub.
- Permisos para que Railway pueda leer el repositorio del proyecto.

---

## 📁 Archivos clave para el despliegue

### 1. `requirements.txt`

Lista de dependencias necesarias para correr el dashboard en producción, por ejemplo:

- `dash`
- `dash-bootstrap-components`
- `plotly`
- `numpy`
- `pandas`
- `nltk`
- `joblib`
- `scikit-learn`

Railway usa este archivo para instalar los paquetes con `pip` durante el proceso de **build**.

### 2. `Procfile`

Archivo de texto sin extensión que indica a Railway cómo arrancar el servicio web:

```text
web: python app_final.py
```
- `web:` indica que es un proceso web.  
- `python app_final.py` ejecuta la aplicación Dash.

### 3. `app_final.py`

Archivo principal de la aplicación Dash:

- Define el layout (logos, textarea, botón, tabla de resultados).  
- Carga el modelo `modelo_cuoc_rf_compacto.pkl`.  
- Expone el servidor en el host y puerto que Railway necesita.

Al final del archivo se debe incluir:

    if __name__ == "__main__":
        import os
        port = int(os.environ.get("PORT", 8050))
        app.run(host="0.0.0.0", port=port, debug=False)

Esto permite:

- **En local:** si `PORT` no está definido, se usa el puerto `8050`.  
- **En la nube (Railway):** se usa el valor de `PORT` que inyecta la plataforma.

---

### 4. `modelo_cuoc_rf_compacto.pkl`

Modelo serializado (pipeline con `CountVectorizer` + `RandomForestClassifier`) que se carga en `app_final.py` para hacer las predicciones de códigos CUOC.

Este archivo **debe existir en la raíz del proyecto** y estar versionado en Git (incluido en el repo).

---

## 🚀 Paso a paso: despliegue en Railway

### 1. Crear el proyecto en Railway

1. Iniciar sesión en Railway.  
2. Hacer clic en **New Project**.  
3. Seleccionar la opción **Deploy from GitHub repo**.  
4. Elegir el repositorio del proyecto (por ejemplo, `vernymendoza/DSA202515_Project`).  

Railway creará un servicio vinculado a la rama `main` del repositorio.

---

### 2. Proceso de build automático

Una vez creado el servicio, Railway (a través de Railpack):

- Detecta que el proyecto es de **Python**.  
- Ejecuta la instalación de paquetes con:

    pip install -r requirements.txt

- Busca un comando de arranque:
  - Revisa el archivo `Procfile`.  
  - Toma la línea `web: python app_final.py` como comando principal.

Si todo es correcto, el build termina con el estado:

- **Deployment successful**

---

### 3. Verificar que el despliegue esté activo

En la pestaña **Deployments** del servicio:

- El último deployment debe estar con estado **ACTIVE**.  
- No debe aparecer `CRASHED` ni `FAILED`.  

Si el servicio está activo, ya se puede exponer al público mediante un dominio.

---

## 🌐 Exponer el servicio (Networking)

Por defecto, el servicio es interno a Railway. Para exponerlo:

1. Ir a la pestaña **Settings** del servicio.  
2. Buscar la sección **Networking**.  
3. En el bloque **Generate Service Domain**:
   - Confirmar el puerto en el que escucha el servicio web (Railway lo maneja internamente usando la variable `PORT`).  
   - Hacer clic en **Generate Domain**.

Railway generará una URL pública similar a:

- `https://dsa202515project-production.up.railway.app`

Con esa URL, cualquier usuario puede acceder al dashboard CUOC desde el navegador.

---

## 🔍 Monitoreo y logs

Railway ofrece dos tipos de logs importantes:

### 1. Build Logs

Se acceden desde la pestaña **Build Logs**.  
Muestran:

- Instalación de dependencias.  
- Errores relacionados con `requirements.txt`.  
- Problemas de compatibilidad de paquetes.

### 2. Runtime / Deploy Logs

Se acceden desde **Deploy Logs** o la sección de **Logs** del deployment activo.  
Muestran:

- Trazas de errores de Python.  
- Errores al cargar el modelo.  
- Excepciones generadas por callbacks de Dash.

---

## 🔁 Actualizar la versión desplegada

Cada vez que se actualiza el proyecto:

1. Realizar cambios locales en el código o el modelo.  
2. Hacer commit y push a la rama `main`:

    git add .
    git commit -m "Actualizo app Dash y modelo CUOC"
    git push origin main

3. Railway detecta el nuevo commit y dispara automáticamente un nuevo deployment.  
4. Si el build y la ejecución son exitosos, la nueva versión reemplaza a la anterior.

---

## 🧪 Checklist de validación en producción

Después de cada despliegue se recomienda:

### Verificar estado del deployment

- Debe estar en estado **ACTIVE / Deployment successful**.

### Probar la URL pública

- Entrar a la URL generada por Railway.  
- Confirmar que:
  - Carga el dashboard.  
  - Se muestran los logos y el título.

### Hacer pruebas funcionales rápidas

- Probar descripción vacía → debe mostrar mensaje de validación.  
- Probar una oferta real → debe mostrar códigos CUOC sugeridos.  
- Probar textos cortos y largos → la app no debe romperse.

### Revisar logs

- Confirmar que no se estén generando errores recurrentes en runtime.

---

## 🚨 Troubleshooting específico de Railway

### Error: `No start command was found`

- Verificar que el archivo `Procfile` exista en la raíz del repositorio.  
- Confirmar que su contenido es exactamente:

    web: python app_final.py

- Hacer `git add Procfile`, `git commit` y `git push` de nuevo.

### Error durante el build (fallas en `requirements.txt`)

- Revisar la pestaña **Build Logs**.  
- Verificar que todos los paquetes estén bien escritos.  
- Si alguna librería no es necesaria en producción, se puede eliminar del `requirements.txt` para simplificar.

### Servicio en estado `CRASHED`

- Revisar los **Deploy Logs**.  
- Errores típicos:
  - `FileNotFoundError` al cargar `modelo_cuoc_rf_compacto.pkl`.  
  - Errores de importación de módulos.  
  - Cambios de nombre en ids de componentes de Dash que rompen callbacks.

---

## 📌 Resumen

- El **dashboard Dash** (`app_final.py`) y el **modelo CUOC** (`modelo_cuoc_rf_compacto.pkl`) se despliegan como un único servicio web en Railway.  
- `requirements.txt` y `Procfile` permiten que Railway:
  - Instale las dependencias necesarias.  
  - Sepa qué comando usar para arrancar el servidor.  
- La variable de entorno `PORT` es gestionada automáticamente por Railway y leída en el código Python.  
- El dominio público se configura desde la sección **Networking** del servicio.
