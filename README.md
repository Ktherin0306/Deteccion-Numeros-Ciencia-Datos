
# Reconocimiento de dígitos escritos a mano con una CNN

Aplicación web que permite escribir un dígito del **0 al 9** en un lienzo y obtener la predicción de una **red neuronal convolucional (CNN)** entrenada con el conjunto de datos MNIST.

El proyecto está desarrollado en Python. Utiliza TensorFlow/Keras para entrenar y ejecutar el modelo, Flask para exponer la API y HTML, CSS y JavaScript para la interfaz de dibujo.

## Características

- Reconocimiento de dígitos manuscritos del 0 al 9.
- Entrenamiento con MNIST, que contiene imágenes de dígitos de 28 × 28 píxeles.
- Aumento de datos mediante pequeñas rotaciones y traslaciones durante el entrenamiento.
- Preprocesamiento que recorta el dígito, conserva su proporción y lo centra en una imagen de 28 × 28 píxeles.
- Interfaz web para dibujar, borrar y reconocer un dígito.
- Visualización de la predicción, su confianza y las tres opciones con mayor probabilidad.
- API con un endpoint de salud para comprobar qué modelo está cargado.
- Prueba automatizada de extremo a extremo para revisar predicciones de los diez dígitos y detectar si el sistema se concentra en una sola clase.

## Tecnologías

- **Python**
- **TensorFlow / Keras** — entrenamiento e inferencia de la CNN.
- **MNIST** — conjunto de datos de dígitos manuscritos.
- **Flask** — servidor web y API.
- **NumPy** — procesamiento de matrices e imágenes.
- **Pillow (PIL)** — lectura y transformación de imágenes.
- **SciPy** — centrado de la imagen por centro de masa.
- **HTML, CSS y JavaScript** — interfaz web.
- **Jupyter Notebook** — ejecución del flujo de trabajo.

## Requisitos

- Python instalado.
- Jupyter Notebook o JupyterLab.
- Conexión a Internet durante la primera ejecución para instalar dependencias y descargar MNIST, si aún no está en caché.
- Memoria y espacio en disco suficientes para entrenar y guardar el modelo.

Las dependencias utilizadas por el notebook son:

```bash
python -m pip install tensorflow flask pillow scipy numpy
```

> El notebook también instala estas dependencias en su primera celda mediante `%pip`. Ejecútalo con el kernel de Python en el que quieras instalar y utilizar los paquetes.

## Cómo ejecutar el proyecto

### 1. Abrir el notebook

Abre `Reconocimiento_Digitos_CNN_MNIST.ipynb` en Jupyter Notebook o JupyterLab. Selecciona el kernel de Python adecuado.

### 2. Ejecutar las celdas en orden

Ejecuta todas las celdas desde el principio, sin saltarte pasos. El notebook crea una carpeta independiente llamada `proyecto_digitos_cnn` en el directorio de trabajo de Jupyter. Dentro de ella genera el backend, los archivos del frontend y los archivos del modelo.

El flujo principal es:

1. Instalar las dependencias y crear las carpetas del proyecto.
2. Definir el preprocesamiento de imágenes.
3. Descargar MNIST, construir y entrenar la CNN.
4. Evaluar el modelo con las imágenes de prueba y guardar los archivos del modelo.
5. Crear la API Flask y la interfaz web.
6. Ejecutar las pruebas automáticas de la API.
7. Iniciar la aplicación y abrirla en el navegador.

La primera ejecución puede tardar más porque debe descargar MNIST y entrenar la red. El entrenamiento utiliza hasta 8 épocas y puede detenerse antes si la exactitud de validación deja de mejorar.

### 3. Abrir la aplicación

Cuando la celda de inicio confirme que el servidor está activo, abre esta dirección en el navegador:

**http://127.0.0.1:5001**

Escribe un solo dígito grande y centrado en el lienzo y pulsa **Reconocer dígito**. Usa **Borrar** para limpiar el lienzo antes de probar otro número.

### 4. Detener el servidor

Al terminar, ejecuta la última celda del notebook, titulada **Detener el servidor**.

## Estructura generada

Al ejecutar las celdas que generan los archivos, la carpeta tendrá una estructura similar a esta:

```text
proyecto_digitos_cnn/
├── backend/
│   ├── app.py              # Servidor Flask y endpoint de predicción
│   ├── preprocess.py       # Preparación de las imágenes
│   ├── digit_cnn.keras     # Modelo CNN entrenado
│   └── model_info.json     # Información y exactitud de prueba
└── frontend/
    ├── templates/
    │   └── index.html      # Página de la aplicación
    └── static/
        ├── app.js          # Dibujo y comunicación con la API
        └── style.css       # Estilos de la interfaz
```

Los archivos del modelo (`digit_cnn.keras` y `model_info.json`) se crean durante el entrenamiento; no estarán disponibles antes de ejecutar esa celda.

## API

La API se ejecuta localmente en el puerto `5001`.

### Comprobar el estado del modelo

Abre esta dirección en el navegador:

```text
http://127.0.0.1:5001/api/health
```

La respuesta indica el estado, el nombre del modelo (`CNN-MNIST`), las clases disponibles del 0 al 9, la ruta del modelo cargado y la exactitud de prueba registrada.

### Realizar una predicción

- **Método:** `POST`
- **Ruta:** `/api/predict`
- **Contenido:** JSON con la propiedad `image`, que debe contener una imagen PNG codificada en base64. También se acepta una URL de datos que empiece con `data:image/png;base64,`.

La respuesta incluye el dígito previsto, la confianza, las tres probabilidades principales y el nombre del modelo.

La API está configurada para escuchar en `127.0.0.1`, por lo que está pensada para pruebas locales en el mismo equipo.

## Cómo se procesan los dibujos

El lienzo del navegador utiliza fondo blanco y trazo oscuro. Antes de predecir, el backend invierte los tonos para que coincidan con el formato de MNIST (fondo negro y dígito claro), detecta el contorno, recorta el trazo, lo ajusta a un área de 20 × 20 píxeles y lo centra en una imagen de 28 × 28. Finalmente, normaliza los valores entre 0 y 1, que es el formato de entrada de la CNN.

## Pruebas y evaluación

El notebook evalúa el modelo con las 10.000 imágenes de prueba de MNIST e imprime la exactitud obtenida. Además, la prueba de extremo a extremo envía a la API ejemplos de los diez dígitos, comprueba que el endpoint de salud indique `CNN-MNIST` y verifica que un lienzo vacío o una imagen no válida produzcan un error controlado.

La exactitud sobre MNIST no garantiza el mismo resultado con todos los estilos de escritura. Es recomendable probar los números escritos en la propia aplicación, procurando dibujarlos grandes, centrados y de uno en uno.

## Solución de problemas

**La aplicación no abre.**

- Comprueba que la última celda de inicio haya confirmado que el servidor está activo.
- Revisa si el puerto `5001` ya está ocupado por otra aplicación.
- Si el servidor se detuvo, vuelve a ejecutar la celda de inicio después de completar el entrenamiento.

**Aparece un error indicando que no existe `digit_cnn.keras`.**

Ejecuta primero la celda de entrenamiento y verifica que haya terminado correctamente. El archivo se guarda en `proyecto_digitos_cnn/backend/digit_cnn.keras`.

**La página sigue usando el modelo antiguo o aparece el puerto 5000.**

Esta versión CNN utiliza el puerto `5001`. Abre `http://127.0.0.1:5001` y comprueba `/api/health`: el campo `model` debe indicar `CNN-MNIST`. No uses la página antigua que esté ejecutándose en el puerto `5000`.

**La predicción no coincide con el número escrito.**

Escribe un único dígito, de tamaño grande y centrado. Consulta las alternativas y sus probabilidades que se muestran debajo de la predicción. Si persiste el problema, ejecuta la celda de prueba de extremo a extremo y revisa los resultados del notebook.

**Falla la instalación de TensorFlow.**

Comprueba la versión de Python de tu entorno y sigue las versiones de Python compatibles con la versión de TensorFlow que pip intenta instalar. Ejecuta la instalación desde el mismo entorno que utiliza el kernel de Jupyter.

## Alcance y limitaciones

- El modelo se entrena con MNIST; la escritura libre de una persona puede diferir de ese conjunto de datos.
- Se espera un dígito por imagen, no palabras ni varios números juntos.
- El proyecto está configurado para uso local y no incluye autenticación ni despliegue público.
- La exactitud real y los resultados de las pruebas deben tomarse de la ejecución del notebook en el equipo donde se entrene el modelo.

## Licencia

No se ha definido una licencia específica para este proyecto. Si vas a publicarlo o compartirlo como software reutilizable, añade una licencia adecuada y revisa los términos aplicables a sus dependencias y al conjunto de datos.
