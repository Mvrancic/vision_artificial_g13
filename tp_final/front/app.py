from __future__ import annotations

import os
import socket
import sys

import gradio_client.utils as gradio_client_utils


TP_FINAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(TP_FINAL_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tp_final.front.demo import build_demo


def patch_gradio_api_info() -> None:
    original_get_type = gradio_client_utils.get_type
    original_json_schema_to_python_type = gradio_client_utils._json_schema_to_python_type

    def safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return original_get_type(schema)

    def safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return original_json_schema_to_python_type(schema, defs)

    gradio_client_utils.get_type = safe_get_type
    gradio_client_utils._json_schema_to_python_type = safe_json_schema_to_python_type


def pick_port(preferred: int = 7860, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    patch_gradio_api_info()
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=pick_port(), show_api=False)


if __name__ == "__main__":
    main()
