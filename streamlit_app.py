"""Reconocimiento de dígitos escritos a mano con una CNN (MNIST) — versión Streamlit."""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

ROOT = Path(__file__).resolve().parent

# Busca la carpeta que contiene el modelo y el preprocesamiento.
CANDIDATES = [ROOT / "proyecto_digitos_cnn" / "backend", ROOT / "backend", ROOT]
BACKEND_DIR = next(
    (p for p in CANDIDATES if (p / "digit_cnn.keras").is_file() and (p / "preprocess.py").is_file()),
    None,
)

st.set_page_config(page_title="Reconocimiento de dígitos con CNN", page_icon="🔢")

if BACKEND_DIR is None:
    st.error(
        "No encuentro `digit_cnn.keras` y `preprocess.py` en el repositorio. "
        "Revisa que ambos archivos estén subidos a GitHub (carpeta `proyecto_digitos_cnn/backend/`)."
    )
    st.stop()

sys.path.insert(0, str(BACKEND_DIR))
from preprocess import to_model_image  # noqa: E402


@st.cache_resource(show_spinner="Cargando la red neuronal…")
def load_model():
    import tensorflow as tf

    return tf.keras.models.load_model(BACKEND_DIR / "digit_cnn.keras")


def canvas_to_gray(image_rgba):
    """RGBA del lienzo -> escala de grises con fondo blanco (255 = blanco, 0 = trazo)."""
    rgba = Image.fromarray(np.asarray(image_rgba, dtype=np.uint8), mode="RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(white, rgba).convert("L"), dtype=np.uint8)


def predict(image_rgba):
    """Devuelve (matriz 28x28 que ve la CNN, probabilidades de 0 a 9) o None si el lienzo está vacío."""
    features = to_model_image(canvas_to_gray(image_rgba))
    if features is None:
        return None
    probabilities = load_model().predict(features[np.newaxis, ...], verbose=0)[0]
    return features[..., 0], probabilities


st.title("¿Qué número escribí?")
st.caption("Red neuronal convolucional (CNN) entrenada con MNIST")
st.write("Escribe un solo número del 0 al 9, **grande y centrado**, con el mouse o el dedo.")

if "canvas_id" not in st.session_state:
    st.session_state.canvas_id = 0

left, right = st.columns([1, 1])
with left:
    canvas = st_canvas(
        stroke_width=25,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=280,
        width=280,
        drawing_mode="freedraw",
        display_toolbar=False,
        return_image_data=True,
        key=f"canvas_{st.session_state.canvas_id}",
    )
    if st.button("Borrar", use_container_width=True):
        st.session_state.canvas_id += 1
        st.rerun()

with right:
    result = None
    if canvas.image_data is not None:
        result = predict(canvas.image_data)
    if result is None:
        st.info("Dibuja un dígito y aquí aparecerá el resultado.")
    else:
        seen_by_model, probabilities = result
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        st.metric("Dígito reconocido", best, f"{probabilities[best]:.1%} de confianza", delta_color="off")
        st.write("Otras opciones: " + " · ".join(f"**{int(i)}** ({probabilities[i]:.1%})" for i in order[1:3]))
        st.bar_chart({"Probabilidad": {str(i): float(p) for i, p in enumerate(probabilities)}})
        with st.expander("Lo que ve la red (28×28)"):
            st.image(np.clip(seen_by_model, 0, 1), width=140, clamp=True)
