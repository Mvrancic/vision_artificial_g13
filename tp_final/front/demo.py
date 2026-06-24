from __future__ import annotations

from pathlib import Path

import gradio as gr

from tp_final.front import utils
from tp_final.front.services import perspective, reconstruction, renovation


def _load_default(path: Path):
    return utils.resize_for_display(utils.load_rgb_image(path), max_width=900, max_height=620)


def _prep(image):
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        return None
    return utils.resize_for_display(rgb, max_width=900, max_height=620)


# --- Perspectiva: click para marcar 4 puntos sobre el piso -----------------

def reset_perspective(image):
    base = _prep(image)
    if base is None:
        return None, [], "Subí o capturá una foto, después marcá 4 puntos sobre el piso."
    return base, [], "0/4 puntos marcados. Click sobre el piso para marcar el primero."


def add_floor_point(evt: gr.SelectData, base, points):
    if base is None:
        raise gr.Error("Subí o capturá una foto antes de marcar puntos.")
    points = list(points or [])
    if len(points) >= 4:
        points = []
    x, y = evt.index
    points.append([int(x), int(y)])
    preview = perspective.annotate_selection(base, points)
    if len(points) < 4:
        status = f"{len(points)}/4 puntos marcados."
    else:
        status = "4/4 puntos. Click en Renderizar (o seguí clickeando para reiniciar la marca)."
    return preview, points, status


def clear_floor(base):
    if base is None:
        return None, [], "Subí o capturá una foto, después marcá 4 puntos sobre el piso."
    return base, [], "0/4 puntos marcados."


def run_perspective(
    base,
    points,
    furniture_name,
    height_ratio,
    offset_x,
    offset_y,
    rotation_deg,
    scale_x,
    scale_y,
    custom_sprite,
):
    if base is None:
        raise gr.Error("Subí o capturá una foto antes de renderizar.")
    image, contact_point = perspective.render_result(
        base,
        points,
        furniture_name,
        float(height_ratio),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        rotation_deg=float(rotation_deg),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        custom_sprite=custom_sprite,
    )
    return image, contact_point


def send_to_reconstruction(image, footprint=None):
    if image is None:
        raise gr.Error("No hay imagen para enviar a Reconstrucción.")
    return image, footprint, gr.Tabs(selected="tab_reconstruccion")


def send_to_perspective(image):
    if image is None:
        raise gr.Error("No hay imagen para enviar a Perspectiva.")
    base, points, status = reset_perspective(image)
    return base, base, points, status, gr.Tabs(selected="tab_perspectiva")


def on_upload_furniture(filepath):
    if not filepath:
        raise gr.Error("No se pudo cargar el PNG del mueble.")
    sprite, warning = perspective.load_custom_sprite(filepath)
    status = warning or "Mueble propio cargado. Click en 'Renderizar' para probarlo."
    return sprite, status


def clear_custom_furniture():
    return None, "Volviste a la lista de muebles predefinidos."


# --- Renovacion: click en 2 esquinas para marcar el objeto a borrar --------

def reset_renovation(image):
    base = _prep(image)
    if base is None:
        return None, [], "Subí o capturá una foto, después click en las 2 esquinas del objeto a borrar."
    return base, [], "0/2 puntos. Click en la esquina superior izquierda del objeto."


def add_renovation_point(evt: gr.SelectData, base, points):
    if base is None:
        raise gr.Error("Subí o capturá una foto antes de marcar la zona a borrar.")
    points = list(points or [])
    if len(points) >= 2:
        points = []
    x, y = evt.index
    points.append([int(x), int(y)])
    preview, _bbox = renovation.annotate_selection(base, points)
    if len(points) < 2:
        status = "1/2 puntos. Click en la esquina inferior derecha del objeto."
    else:
        status = "ROI lista. Click en 'Eliminar objeto' (o seguí clickeando para reiniciar la selección)."
    return preview, points, status


def clear_renovation_selection(base):
    if base is None:
        return None, [], "Subí o capturá una foto, después click en las 2 esquinas del objeto a borrar."
    return base, [], "0/2 puntos. Click en la esquina superior izquierda del objeto."


def update_renovation_backend(method_name):
    return f"Backend seleccionado: {method_name}. {renovation.get_backend_status(method_name)}"


def run_renovation_with_controls(base, points, radius, method_name, dilate_iter, expand_px, feather_px):
    if base is None:
        raise gr.Error("Subí o capturá una foto antes de eliminar el objeto.")
    return renovation.run_renovation_from_points(
        base,
        points,
        int(radius),
        method_name,
        int(dilate_iter),
        int(expand_px),
        int(feather_px),
    )


# --- Reconstruccion ----------------------------------------------------------

def run_reconstruction(image, near_pct, plan_mode, footprint):
    return reconstruction.process_reconstruction(image, int(near_pct), plan_mode, synthetic_point=footprint)


def clear_reconstruction_footprint():
    return None


# --- Subida de foto explicita (boton, no depende de "limpiar" el Image) -----

def on_upload_perspective(filepath):
    if not filepath:
        raise gr.Error("No se pudo cargar la imagen.")
    image = utils.load_rgb_image(filepath)
    base, points, status = reset_perspective(image)
    return base, base, points, status


def on_upload_renovation(filepath):
    if not filepath:
        raise gr.Error("No se pudo cargar la imagen.")
    image = utils.load_rgb_image(filepath)
    base, points, status = reset_renovation(image)
    return base, base, points, status


def on_upload_reconstruction(filepath):
    if not filepath:
        raise gr.Error("No se pudo cargar la imagen.")
    image = utils.load_rgb_image(filepath)
    return _prep(image)


def build_demo() -> gr.Blocks:
    default_perspective = _load_default(utils.PERSPECTIVE_SAMPLE)
    default_renovation = _load_default(utils.RENOVATION_SAMPLE)
    default_reconstruction = _load_default(utils.RECONSTRUCTION_SAMPLE)

    with gr.Blocks(title="TP Final - Interiorismo con Vision Artificial") as demo:
        gr.Markdown(
            """
            # TP Final: Interiorismo con Vision Artificial
            Trabajá con los tres enfoques desde una sola interfaz: insertar muebles en perspectiva,
            eliminar objetos con inpainting y reconstruir una planta aproximada a partir de una foto.
            Para marcar puntos sobre una imagen, **hacé click directamente sobre ella** (no hay que pintar).
            """
        )

        with gr.Tabs() as tabs:
            with gr.Tab("Perspectiva", id="tab_perspectiva"):
                gr.Markdown("Subí una foto y hacé **click** sobre 4 puntos del piso (las 4 esquinas donde apoyaría el mueble).")
                perspective_base = gr.State(default_perspective)
                perspective_points = gr.State([])
                custom_furniture = gr.State(None)
                perspective_footprint = gr.State(None)
                perspective_upload_btn = gr.UploadButton(
                    "📁 Subir foto", file_types=["image"], variant="primary"
                )
                perspective_canvas = gr.Image(
                    value=default_perspective,
                    sources=["upload"],
                    type="numpy",
                    label="Imagen del ambiente (o usá el botón de arriba para subir la tuya)",
                )
                with gr.Row():
                    clear_floor_btn = gr.Button("Limpiar puntos")
                    render_perspective_btn = gr.Button("Renderizar", variant="primary")
                with gr.Row():
                    with gr.Column(scale=1):
                        furniture_name = gr.Dropdown(
                            choices=perspective.furniture_names(),
                            value=perspective.furniture_names()[0],
                            label="Mueble (de la lista)",
                        )
                        furniture_upload_btn = gr.UploadButton(
                            "📁 Subir mi mueble (PNG transparente)", file_types=["image"]
                        )
                        clear_furniture_btn = gr.Button("Volver a la lista de muebles")
                    with gr.Column(scale=1):
                        height_ratio = gr.Slider(0.3, 1.8, value=0.8, step=0.05, label="Altura relativa")
                        offset_x = gr.Slider(-200, 200, value=0, step=2, label="Mover horizontal (px)")
                        offset_y = gr.Slider(-200, 200, value=0, step=2, label="Mover vertical (px)")
                        rotation_deg = gr.Slider(-45, 45, value=0, step=1, label="Rotar (°)")
                    with gr.Column(scale=1):
                        scale_x = gr.Slider(0.3, 2.5, value=1.0, step=0.05, label="Ancho")
                        scale_y = gr.Slider(0.3, 2.5, value=1.0, step=0.05, label="Alto")
                with gr.Row():
                    perspective_status = gr.Markdown(
                        "0/4 puntos marcados. Click sobre el piso para marcar el primero."
                    )
                perspective_result = gr.Image(type="numpy", label="Resultado", buttons=["download", "fullscreen"])
                send_perspective_to_reconstruction_btn = gr.Button("📐 Usar esta imagen en Reconstrucción")

                perspective_canvas.upload(
                    reset_perspective,
                    inputs=[perspective_canvas],
                    outputs=[perspective_base, perspective_points, perspective_status],
                    api_name="perspective_reset_on_upload",
                )
                perspective_canvas.select(
                    add_floor_point,
                    inputs=[perspective_base, perspective_points],
                    outputs=[perspective_canvas, perspective_points, perspective_status],
                    api_name="perspective_add_point",
                )
                clear_floor_btn.click(
                    clear_floor,
                    inputs=[perspective_base],
                    outputs=[perspective_canvas, perspective_points, perspective_status],
                    api_name="perspective_clear_points",
                )
                perspective_render_inputs = [
                    perspective_base,
                    perspective_points,
                    furniture_name,
                    height_ratio,
                    offset_x,
                    offset_y,
                    rotation_deg,
                    scale_x,
                    scale_y,
                    custom_furniture,
                ]
                perspective_render_outputs = [perspective_result, perspective_footprint]
                render_perspective_btn.click(
                    run_perspective,
                    inputs=perspective_render_inputs,
                    outputs=perspective_render_outputs,
                    api_name="perspective_render",
                )
                offset_x.change(
                    run_perspective,
                    inputs=perspective_render_inputs,
                    outputs=perspective_render_outputs,
                    api_name="perspective_move_x",
                )
                offset_y.change(
                    run_perspective,
                    inputs=perspective_render_inputs,
                    outputs=perspective_render_outputs,
                    api_name="perspective_move_y",
                )
                rotation_deg.change(
                    run_perspective,
                    inputs=perspective_render_inputs,
                    outputs=perspective_render_outputs,
                    api_name="perspective_rotate",
                )
                scale_x.change(
                    run_perspective,
                    inputs=perspective_render_inputs,
                    outputs=perspective_render_outputs,
                    api_name="perspective_scale_width",
                )
                scale_y.change(
                    run_perspective,
                    inputs=perspective_render_inputs,
                    outputs=perspective_render_outputs,
                    api_name="perspective_scale_height",
                )
                perspective_upload_btn.upload(
                    on_upload_perspective,
                    inputs=[perspective_upload_btn],
                    outputs=[perspective_canvas, perspective_base, perspective_points, perspective_status],
                    api_name="perspective_upload_button",
                )
                furniture_upload_btn.upload(
                    on_upload_furniture,
                    inputs=[furniture_upload_btn],
                    outputs=[custom_furniture, perspective_status],
                    api_name="perspective_upload_furniture",
                )
                clear_furniture_btn.click(
                    clear_custom_furniture,
                    outputs=[custom_furniture, perspective_status],
                    api_name="perspective_clear_furniture",
                )

            with gr.Tab("Renovacion", id="tab_renovacion"):
                gr.Markdown(
                    "Subí una foto y hacé **click en 2 esquinas** (superior izquierda e inferior derecha) "
                    "del objeto que querés borrar. La segmentación del objeto dentro de esa zona es automática."
                )
                renovation_base = gr.State(default_renovation)
                renovation_points = gr.State([])
                renovation_upload_btn = gr.UploadButton(
                    "📁 Subir foto", file_types=["image"], variant="primary"
                )
                renovation_canvas = gr.Image(
                    value=default_renovation,
                    sources=["upload"],
                    type="numpy",
                    label="Imagen a limpiar (o usá el botón de arriba para subir la tuya)",
                )
                with gr.Row():
                    clear_mask_btn = gr.Button("Limpiar seleccion")
                    run_renovation_btn = gr.Button("Eliminar objeto", variant="primary")
                renovation_status = gr.Markdown(
                    "0/2 puntos. Click en la esquina superior izquierda del objeto."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        method_name = gr.Radio(
                            choices=["TELEA", "Navier-Stokes", "LaMa"],
                            value="TELEA",
                            label="Metodo de inpainting",
                        )
                    with gr.Column(scale=1):
                        radius = gr.Slider(1, 20, value=5, step=1, label="Radio de inpaint")
                        dilate_iter = gr.Slider(1, 15, value=4, step=1, label="Dilatacion")
                    with gr.Column(scale=1):
                        expand_px = gr.Slider(0, 40, value=12, step=1, label="Expansion de mascara")
                        feather_px = gr.Slider(0, 20, value=6, step=1, label="Suavizado de borde")
                with gr.Row():
                    renovation_masked = gr.Image(type="numpy", label="Mascara aplicada")
                    renovation_mask = gr.Image(type="numpy", label="Mascara binaria")
                    renovation_result = gr.Image(
                        type="numpy", label="Resultado limpio", buttons=["download", "fullscreen"]
                    )
                with gr.Row():
                    send_renovation_to_perspective_btn = gr.Button("🛋️ Usar esta imagen en Perspectiva")
                    send_renovation_to_reconstruction_btn = gr.Button("📐 Usar esta imagen en Reconstrucción")

                renovation_canvas.upload(
                    reset_renovation,
                    inputs=[renovation_canvas],
                    outputs=[renovation_base, renovation_points, renovation_status],
                    api_name="renovation_reset_on_upload",
                )
                renovation_canvas.select(
                    add_renovation_point,
                    inputs=[renovation_base, renovation_points],
                    outputs=[renovation_canvas, renovation_points, renovation_status],
                    api_name="renovation_add_point",
                )
                clear_mask_btn.click(
                    clear_renovation_selection,
                    inputs=[renovation_base],
                    outputs=[renovation_canvas, renovation_points, renovation_status],
                    api_name="renovation_clear_selection",
                )
                method_name.change(
                    update_renovation_backend,
                    inputs=[method_name],
                    outputs=[renovation_status],
                    api_name="renovation_backend_status",
                )
                run_renovation_btn.click(
                    run_renovation_with_controls,
                    inputs=[
                        renovation_base,
                        renovation_points,
                        radius,
                        method_name,
                        dilate_iter,
                        expand_px,
                        feather_px,
                    ],
                    outputs=[renovation_masked, renovation_result, renovation_mask],
                    api_name="renovation_run",
                )
                renovation_upload_btn.upload(
                    on_upload_renovation,
                    inputs=[renovation_upload_btn],
                    outputs=[renovation_canvas, renovation_base, renovation_points, renovation_status],
                    api_name="renovation_upload_button",
                )

            with gr.Tab("Reconstruccion", id="tab_reconstruccion"):
                reconstruction_footprint = gr.State(None)
                gr.Markdown(
                    "Si la foto viene de 'Usar esta imagen en Reconstrucción' desde Perspectiva, "
                    "el mueble agregado se marca en el plano con un diamante magenta (posición conocida, "
                    "no detectada por profundidad). Si subís una foto nueva acá, esa marca se borra."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        reconstruction_upload_btn = gr.UploadButton(
                            "📁 Subir foto", file_types=["image"], variant="primary"
                        )
                        reconstruction_input = gr.Image(
                            value=default_reconstruction,
                            sources=["upload"],
                            type="numpy",
                            label="Imagen de escena (o usá el botón de arriba para subir la tuya)",
                        )
                    with gr.Column(scale=3):
                        near_pct = gr.Slider(60, 99, value=80, step=1, label="Umbral de cercania (%)")
                        plan_mode = gr.Radio(
                            choices=["heat", "color"],
                            value="heat",
                            label="Visualizacion de planta",
                        )
                        run_reconstruction_btn = gr.Button("Reconstruir escena", variant="primary")
                        reconstruction_status = gr.Markdown("Procesá una imagen para ver profundidad, objetos y planta.")
                reconstruction_summary = gr.Image(type="numpy", label="Resumen 2x2")
                with gr.Row():
                    reconstruction_original = gr.Image(type="numpy", label="Original")
                    reconstruction_depth = gr.Image(type="numpy", label="Profundidad")
                with gr.Row():
                    reconstruction_objects = gr.Image(type="numpy", label="Objetos detectados")
                    reconstruction_plan = gr.Image(type="numpy", label="Planta")

                run_reconstruction_btn.click(
                    run_reconstruction,
                    inputs=[reconstruction_input, near_pct, plan_mode, reconstruction_footprint],
                    outputs=[
                        reconstruction_summary,
                        reconstruction_original,
                        reconstruction_depth,
                        reconstruction_objects,
                        reconstruction_plan,
                        reconstruction_status,
                    ],
                    api_name="reconstruction_run",
                )
                reconstruction_upload_btn.upload(
                    on_upload_reconstruction,
                    inputs=[reconstruction_upload_btn],
                    outputs=[reconstruction_input],
                    api_name="reconstruction_upload_button",
                )
                reconstruction_upload_btn.upload(
                    clear_reconstruction_footprint,
                    outputs=[reconstruction_footprint],
                    api_name="reconstruction_clear_footprint_on_upload",
                )
                reconstruction_input.upload(
                    clear_reconstruction_footprint,
                    outputs=[reconstruction_footprint],
                    api_name="reconstruction_clear_footprint_on_drop",
                )

            send_perspective_to_reconstruction_btn.click(
                send_to_reconstruction,
                inputs=[perspective_result, perspective_footprint],
                outputs=[reconstruction_input, reconstruction_footprint, tabs],
                api_name="perspective_send_to_reconstruction",
            )
            send_renovation_to_reconstruction_btn.click(
                send_to_reconstruction,
                inputs=[renovation_result],
                outputs=[reconstruction_input, reconstruction_footprint, tabs],
                api_name="renovation_send_to_reconstruction",
            )
            send_renovation_to_perspective_btn.click(
                send_to_perspective,
                inputs=[renovation_result],
                outputs=[perspective_canvas, perspective_base, perspective_points, perspective_status, tabs],
                api_name="renovation_send_to_perspective",
            )

    return demo
