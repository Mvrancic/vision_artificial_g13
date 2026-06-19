from __future__ import annotations

from pathlib import Path

import gradio as gr

from tp_final.front import utils
from tp_final.front.services import perspective, reconstruction, renovation


def _load_default(path: Path):
    return utils.resize_for_display(utils.load_rgb_image(path), max_width=900, max_height=620)


def preview_perspective_selection(editor_value):
    return perspective.annotate_editor_selection(editor_value)


def clear_perspective_outputs():
    return None, None, "Pintá 4 puntos sobre el piso y después previsualizá o renderizá."


def normalize_editor_image(editor_value):
    if not isinstance(editor_value, dict):
        return editor_value
    background = utils.ensure_rgb(editor_value.get("background"))
    if background is None:
        return editor_value
    resized = utils.resize_for_display(background, max_width=900, max_height=620)
    return {"background": resized, "layers": [], "composite": resized}


def normalize_uploaded_image(image):
    rgb = utils.ensure_rgb(image)
    if rgb is None:
        return None
    resized = utils.resize_for_display(rgb, max_width=900, max_height=620)
    return {"background": resized, "layers": [], "composite": resized}


def run_perspective(editor_value, furniture_name, height_ratio, shadow_opacity, show_shadow):
    return perspective.render_from_editor(
        editor_value,
        furniture_name,
        float(height_ratio),
        float(shadow_opacity),
        bool(show_shadow),
    )


def update_renovation_backend(method_name):
    return f"Backend seleccionado: {method_name}. {renovation.get_backend_status(method_name)}"


def clear_renovation_outputs():
    status = "Pintá con el pincel sobre el objeto a borrar y después ejecutá el inpainting."
    return None, None, None, status


def run_renovation_with_controls(
    editor_value,
    radius,
    method_name,
    dilate_iter,
    expand_px,
    feather_px,
):
    return renovation.run_renovation(
        editor_value,
        int(radius),
        method_name,
        int(dilate_iter),
        int(expand_px),
        int(feather_px),
    )


def run_reconstruction(image, near_pct, plan_mode):
    return reconstruction.process_reconstruction(image, int(near_pct), plan_mode)


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
            """
        )

        with gr.Tabs():
            with gr.Tab("Perspectiva"):
                perspective_upload = gr.Image(
                    value=default_perspective,
                    type="numpy",
                    label="Subí la foto del ambiente",
                )
                perspective_editor = gr.ImageEditor(
                    value={"background": default_perspective, "layers": [], "composite": default_perspective},
                    type="numpy",
                    label="Cargá la imagen y marcá 4 puntos sobre el piso",
                    brush=gr.Brush(colors=["#39ff88"], default_color="#39ff88", color_mode="fixed", default_size=22),
                    eraser=gr.Eraser(default_size=28),
                    layers=True,
                    transforms=(),
                    image_mode="RGBA",
                    height=560,
                )
                with gr.Row():
                    clear_floor_btn = gr.Button("Limpiar marcas")
                    preview_floor_btn = gr.Button("Previsualizar piso")
                    render_perspective_btn = gr.Button("Renderizar", variant="primary")
                with gr.Row():
                    with gr.Column(scale=1):
                        furniture_name = gr.Dropdown(
                            choices=perspective.furniture_names(),
                            value=perspective.furniture_names()[0],
                            label="Mueble",
                        )
                    with gr.Column(scale=1):
                        height_ratio = gr.Slider(0.3, 1.8, value=0.8, step=0.05, label="Altura relativa")
                        shadow_opacity = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Opacidad de sombra")
                        show_shadow = gr.Checkbox(value=True, label="Mostrar sombra")
                    with gr.Column(scale=1):
                        perspective_status = gr.Markdown(
                            "Pintá 4 puntos sobre el piso y después previsualizá o renderizá."
                        )
                perspective_preview = gr.Image(
                    type="numpy",
                    label="Seleccion del piso",
                    interactive=False,
                )
                perspective_result = gr.Image(type="numpy", label="Resultado")

                perspective_upload.change(
                    normalize_uploaded_image,
                    inputs=[perspective_upload],
                    outputs=[perspective_editor],
                    api_name="perspective_load_editor",
                )
                preview_floor_btn.click(
                    preview_perspective_selection,
                    inputs=[perspective_editor],
                    outputs=[perspective_preview, perspective_status],
                    api_name="perspective_preview_floor",
                )
                perspective_editor.change(
                    clear_perspective_outputs,
                    outputs=[perspective_preview, perspective_result, perspective_status],
                    api_name="perspective_reset_outputs",
                )
                clear_floor_btn.click(
                    lambda editor: {"background": editor["background"], "layers": [], "composite": editor["background"]} if editor and editor.get("background") is not None else None,
                    inputs=[perspective_editor],
                    outputs=[perspective_editor],
                    api_name="perspective_clear_marks",
                )
                clear_floor_btn.click(
                    clear_perspective_outputs,
                    outputs=[perspective_preview, perspective_result, perspective_status],
                    api_name="perspective_clear_outputs",
                )
                render_perspective_btn.click(
                    run_perspective,
                    inputs=[
                        perspective_editor,
                        furniture_name,
                        height_ratio,
                        shadow_opacity,
                        show_shadow,
                    ],
                    outputs=[perspective_result],
                    api_name="perspective_render",
                )

            with gr.Tab("Renovacion"):
                renovation_upload = gr.Image(
                    value=default_renovation,
                    type="numpy",
                    label="Subí la foto a limpiar",
                )
                renovation_editor = gr.ImageEditor(
                    value={"background": default_renovation, "layers": [], "composite": default_renovation},
                    type="numpy",
                    label="Cargá la imagen y pintá sobre el objeto a borrar",
                    brush=gr.Brush(colors=["#ff4d4f"], default_color="#ff4d4f", color_mode="fixed", default_size=28),
                    eraser=gr.Eraser(default_size=32),
                    layers=True,
                    transforms=(),
                    image_mode="RGBA",
                    height=560,
                )
                with gr.Row():
                    clear_mask_btn = gr.Button("Limpiar mascara")
                    run_renovation_btn = gr.Button("Eliminar objeto", variant="primary")
                renovation_status = gr.Markdown(
                    "Pintá con el pincel sobre el objeto a borrar y después ejecutá el inpainting."
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
                    renovation_result = gr.Image(type="numpy", label="Resultado limpio")

                renovation_upload.change(
                    normalize_uploaded_image,
                    inputs=[renovation_upload],
                    outputs=[renovation_editor],
                    api_name="renovation_load_editor",
                )
                clear_mask_btn.click(
                    lambda editor: {"background": editor["background"], "layers": [], "composite": editor["background"]} if editor and editor.get("background") is not None else None,
                    inputs=[renovation_editor],
                    outputs=[renovation_editor],
                    api_name="renovation_clear_mask",
                )
                clear_mask_btn.click(
                    clear_renovation_outputs,
                    outputs=[renovation_masked, renovation_mask, renovation_result, renovation_status],
                    api_name="renovation_clear_outputs",
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
                        renovation_editor,
                        radius,
                        method_name,
                        dilate_iter,
                        expand_px,
                        feather_px,
                    ],
                    outputs=[renovation_masked, renovation_result, renovation_mask],
                    api_name="renovation_run",
                )

            with gr.Tab("Reconstruccion"):
                with gr.Row():
                    with gr.Column(scale=5):
                        reconstruction_input = gr.Image(
                            value=default_reconstruction,
                            type="numpy",
                            label="Imagen de escena",
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
                    inputs=[reconstruction_input, near_pct, plan_mode],
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

    return demo
