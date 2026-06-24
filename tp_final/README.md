# TP Final — Visión Artificial aplicada al Diseño de Interiores

Aplicación de visión por computadora para el interiorismo, con los **tres enfoques**
de la propuesta. Todo corre con **OpenCV clásico** (+ scikit-learn ya presente en el
repo), sin necesidad de PyTorch/TensorFlow. Interfaz en ventanas OpenCV con
trackbars y mouse, consistente con `tp_deteccion`.

## Cómo correr

`tp_final` necesita **Python 3.10+** (por `simple-lama-inpainting`, `rembg` y
`torch`). El venv del resto del repo es Python 3.9, así que `tp_final` usa su
**propio venv** en `tp_final/.venv`.

```bash
# una sola vez, desde tp_final/
python3.12 -m venv .venv   # o python3.11/python3.10
source .venv/bin/activate
pip install -r requirements.txt

# version de consola (ventanas OpenCV)
python app.py
```

## Frontend web

Se agregó una interfaz en Gradio dentro de `tp_final/front/` para usar los tres
enfoques desde el navegador.

```bash
# desde tp_final/, con el venv de arriba activado
python front/app.py
```

La app levanta un servidor local en `http://127.0.0.1:7860` o en el siguiente
puerto disponible.

`app.py` muestra un menú para elegir enfoque y fuente (imagen del repo o cámara).
Cada enfoque también se puede correr directo:

```bash
python enfoque1_perspectiva/main.py
python enfoque2_renovacion/main.py
python enfoque3_reconstruccion/main.py
```

## Enfoques

### 1. Visualización de muebles en perspectiva real (`enfoque1_perspectiva/`)
El usuario marca **4 puntos en el piso** y la app proyecta un mueble adaptando su
escala y ángulo a la perspectiva de la habitación, respetando las líneas de fuga.
- **Técnica:** homografía plano→perspectiva (`getPerspectiveTransform` +
  `warpPerspective`) sobre un sprite con canal alfa; el mueble se "para" sobre la
  arista trasera del piso (`standing_quad_from_floor`).
- **Sombra sintética:** se estima la dirección de la luz a partir del centroide de
  brillo de la imagen y se proyecta una sombra elipsoidal desplazada en el piso.
- **Controles:** trackbars de altura y opacidad de sombra; `n` cambia de mueble,
  `r` re-marca el piso, `s` guarda, `q` sale.

### 2. Renovación inteligente: borrar + re-decorar (`enfoque2_renovacion/`)
El usuario selecciona un mueble; la app detecta sus bordes, lo elimina y
**rellena** la pared/piso ocultos.
- **Técnica:** `selectROI` + `grabCut` (de `practica_segmentacion`) para una máscara
  fina, refinado morfológico e **inpainting** con dos familias de backend:
  `cv2.inpaint` (TELEA / Navier-Stokes) y **LaMa** para relleno más robusto.
- **Controles:** trackbars de radio de inpaint y dilatación; `m` cambia el método,
  `r` re-selecciona, `s` guarda. Vista comparativa (máscara | resultado limpio).

### 3. Reconstructor de ambientes y planos 3D (`enfoque3_reconstruccion/`)
A partir de una sola foto se estima la profundidad de la escena, se detectan los
muebles y se arma una **vista de planta (top-down)**.
- **Técnica:** profundidad monocular **MiDaS small** corriendo sobre `cv2.dnn`
  (ONNX). Detección de objetos por blobs de cercanía y estimación de volumen
  aproximado. Back-proyección pinhole de los píxeles a coordenadas mundo (X, Z)
  para la planta.
- **Modelo:** `model-small.onnx` se **descarga bajo demanda** a
  `enfoque3_reconstruccion/models/` (no se versiona). Sin red, cae a una
  profundidad heurística para que el demo siga funcionando.
- **Controles:** trackbar de umbral de cercanía; `s` guarda la grilla 2×2
  (original | profundidad | objetos | planta).

## Estructura

```
tp_final/
  app.py                      menú lanzador
  common/                     helpers compartidos
    geometry.py               homografía, compositing alfa, sombra, luz
    point_picker.py           selector de N puntos con mouse
    assets.py                 sprites de muebles (generados por código)
    sources.py                cámara / imágenes del repo
    trackbar.py               trackbars (estilo tp_deteccion)
  enfoque1_perspectiva/
  enfoque2_renovacion/
  enfoque3_reconstruccion/    + midas.py (estimador de profundidad)
  assets/                     PNG de muebles con alfa
```
