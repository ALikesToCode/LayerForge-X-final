from __future__ import annotations

import base64
import argparse
import json
import mimetypes
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import load_config
from .editability import export_target_assets
from .frontier import run_frontier_comparison
from .pipeline import LayerForgePipeline
from .site_data import build_project_site_payload
from .transparent import export_transparent_assets
from .utils import ensure_dir, safe_name


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
JOB_LOCK = threading.Lock()
PIPELINE_CACHE: dict[tuple[str, str], LayerForgePipeline] = {}
PIPELINE_CACHE_LOCK = threading.Lock()


def _parse_prompts(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_point(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    x, y = [int(part.strip()) for part in str(text).split(",", maxsplit=1)]
    return x, y


def _parse_box(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    parts = [int(part.strip()) for part in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("box expects x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace_url(path: Path) -> str:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if root in resolved.parents or resolved == root:
        rel = resolved.relative_to(root)
        return f"/workspace/{rel.as_posix()}"
    return resolved.as_uri()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if root in resolved.parents or resolved == root:
        return str(resolved.relative_to(root))
    return str(resolved)


def _inject_runtime_marker(html: str) -> str:
    marker = '<script>window.__LAYERFORGE_RUNTIME__ = true;</script>'
    if marker in html:
        return html
    return html.replace("</head>", f"    {marker}\n    </head>", 1)


def _resolve_workspace_path(url_path: str) -> Path:
    decoded = unquote(url_path.removeprefix("/workspace/"))
    candidate = (REPO_ROOT / decoded).resolve()
    root = REPO_ROOT.resolve()
    if root not in candidate.parents and candidate != root:
        raise PermissionError("Requested path escapes repository root")
    return candidate


def _get_pipeline(config_path: str, device: str) -> LayerForgePipeline:
    resolved_config = str((REPO_ROOT / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path).resolve())
    key = (resolved_config, device)
    with PIPELINE_CACHE_LOCK:
        pipeline = PIPELINE_CACHE.get(key)
        if pipeline is None:
            cfg = load_config(resolved_config)
            pipeline = LayerForgePipeline(cfg, device=device)
            PIPELINE_CACHE[key] = pipeline
    return pipeline


def _write_uploaded_image(work_root: Path, filename: str, image_base64: str) -> Path:
    uploads_dir = ensure_dir(work_root / "uploads")
    stem = safe_name(Path(filename).stem or "image")
    suffix = Path(filename).suffix.lower() or ".png"
    output = uploads_dir / f"{int(time.time())}_{stem}{suffix}"
    output.write_bytes(base64.b64decode(image_base64.encode("utf-8")))
    return output


def _collect_previews(output_dir: Path, mode: str) -> list[dict[str, str]]:
    preview_specs = [
        ("Input", output_dir / "debug" / "input_rgb.png"),
        ("Segmentation overlay", output_dir / "debug" / "segmentation_overlay.png"),
        ("Ordered layer stack", output_dir / "debug" / "ordered_layer_contact_sheet.png"),
        ("Grouped layer stack", output_dir / "debug" / "grouped_layer_contact_sheet.png"),
        ("Recomposition", output_dir / "debug" / "recomposed_rgb.png"),
        ("Edit remove", output_dir / "debug" / "edit_remove.png"),
        ("Edit move", output_dir / "debug" / "edit_move.png"),
        ("Edit recolor", output_dir / "debug" / "edit_recolor.png"),
    ]
    if mode == "extract":
        preview_specs.extend(
            [
                ("Target RGBA", output_dir / "target_extract" / "target_rgba.png"),
                ("Completed background", output_dir / "target_extract" / "background_completed.png"),
                ("Move preview", output_dir / "target_extract" / "edit_preview_move.png"),
            ]
        )
    if mode == "transparent":
        preview_specs.extend(
            [
                ("Transparent foreground", output_dir / "transparent_extract" / "transparent_foreground_rgba.png"),
                ("Estimated clean background", output_dir / "transparent_extract" / "estimated_clean_background.png"),
                ("Transparent recomposition", output_dir / "transparent_extract" / "recomposition.png"),
            ]
        )
    if mode == "peel":
        preview_specs.extend(
            [
                ("Peeling strip", output_dir / "debug" / "peeling_strip.png"),
            ]
        )
    previews: list[dict[str, str]] = []
    for label, path in preview_specs:
        if path.exists():
            previews.append({"label": label, "url": _workspace_url(path)})
    return previews


def _collect_summary_metrics(output_dir: Path, mode: str, *, frontier_selection: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    summary: dict[str, Any] = {
        "num_layers": metrics.get("num_layers"),
        "recompose_psnr": metrics.get("recompose_psnr"),
        "recompose_ssim": metrics.get("recompose_ssim"),
        "ordering_method": metrics.get("ordering_method"),
        "segmentation_method": metrics.get("segmentation_method"),
        "depth_method": metrics.get("depth_method"),
    }
    if mode == "extract":
        target_meta = output_dir / "target_extract" / "target_metadata.json"
        if target_meta.exists():
            meta = _read_json(target_meta)
            summary["selected_target"] = meta.get("selected_target")
    if mode == "transparent":
        transparent_metrics = output_dir / "transparent_extract" / "transparent_metrics.json"
        if transparent_metrics.exists():
            meta = _read_json(transparent_metrics)
            summary["transparent_alpha_nonzero_ratio"] = meta.get("alpha_nonzero_ratio")
            summary["transparent_recompose_psnr"] = meta.get("recompose_psnr")
            summary["selected_target"] = meta.get("selected_target")
    if mode == "frontier" and frontier_selection:
        summary["selected_label"] = frontier_selection.get("label")
        summary["selected_self_eval_score"] = frontier_selection.get("self_eval_score")
    return {key: value for key, value in summary.items() if value is not None}


def _resolve_run_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidate = (REPO_ROOT / path).resolve()
    return candidate


def run_webui_job(repo_root: Path, payload: dict[str, Any], *, work_root: Path | None = None) -> dict[str, Any]:
    mode = str(payload.get("mode", "run"))
    filename = str(payload.get("filename", "image.png"))
    image_base64 = str(payload.get("image_base64", ""))
    if not image_base64:
        raise ValueError("Missing image payload")

    base_work_root = work_root if work_root is not None else repo_root / "runs" / "webui"
    ensure_dir(base_work_root)
    input_path = _write_uploaded_image(base_work_root, filename, image_base64)
    job_dir = ensure_dir(base_work_root / "jobs" / f"{int(time.time())}_{safe_name(mode)}_{safe_name(input_path.stem)}")

    config_path = str(payload.get("config") or "configs/fast.yaml")
    device = str(payload.get("device") or "auto")
    segmenter = payload.get("segmenter")
    depth_method = payload.get("depth")
    prompt_text = payload.get("prompt")
    prompt_source = payload.get("prompt_source")
    ordering = payload.get("ordering")
    ranker_model = payload.get("ranker_model")
    point = _parse_point(payload.get("point"))
    box = _parse_box(payload.get("box"))
    target_name = payload.get("target_name")
    prompts = _parse_prompts(prompt_text)
    no_parallax = bool(payload.get("no_parallax", True))
    max_layers = payload.get("max_layers")

    frontier_selection: dict[str, Any] | None = None

    with JOB_LOCK:
        if mode == "run":
            pipeline = _get_pipeline(config_path, device)
            out = pipeline.run(
                input_path,
                job_dir,
                segmenter=segmenter,
                depth_method=depth_method,
                prompts=prompts,
                prompt_source=prompt_source,
                flip_depth=bool(payload.get("flip_depth", False)),
                save_parallax=not no_parallax,
                ordering_method=ordering,
                ranker_model_path=ranker_model,
            )
        elif mode == "extract":
            pipeline = _get_pipeline(config_path, device)
            out = pipeline.run(
                input_path,
                job_dir,
                segmenter=segmenter,
                depth_method=depth_method,
                prompts=prompts,
                prompt_source=prompt_source,
                flip_depth=bool(payload.get("flip_depth", False)),
                save_parallax=not no_parallax,
                ordering_method=ordering,
                ranker_model_path=ranker_model,
            )
            export_target_assets(
                out.output_dir,
                output_dir=job_dir / "target_extract",
                prompt=prompt_text,
                point=point,
                box=box,
                target_name=target_name,
            )
        elif mode == "transparent":
            pipeline = _get_pipeline(config_path, device)
            out = pipeline.run(
                input_path,
                job_dir,
                segmenter=segmenter,
                depth_method=depth_method,
                prompts=prompts,
                prompt_source=prompt_source,
                flip_depth=bool(payload.get("flip_depth", False)),
                save_parallax=False,
                ordering_method=ordering,
                ranker_model_path=ranker_model,
            )
            export_transparent_assets(
                out.output_dir,
                output_dir=job_dir / "transparent_extract",
                prompt=prompt_text,
                point=point,
                box=box,
                target_name=target_name,
            )
        elif mode == "peel":
            pipeline = _get_pipeline(config_path, device)
            out = pipeline.peel(
                input_path,
                job_dir,
                segmenter=segmenter,
                depth_method=depth_method,
                prompts=prompts,
                prompt_source=prompt_source,
                flip_depth=bool(payload.get("flip_depth", False)),
                max_layers=int(max_layers) if max_layers else None,
            )
        elif mode == "frontier":
            frontier_root = job_dir / "frontier"
            frontier_args = argparse.Namespace(
                input_dir=None,
                inputs=[str(input_path)],
                output_root=str(frontier_root),
                native_config=config_path,
                native_segmenter=segmenter or "grounded_sam2",
                native_depth=depth_method or "ensemble",
                skip_native=False,
                peeling_config=str(payload.get("peeling_config") or "configs/recursive_peeling.yaml"),
                peeling_segmenter=str(payload.get("peeling_segmenter") or segmenter or "grounded_sam2"),
                peeling_depth=str(payload.get("peeling_depth") or depth_method or "ensemble"),
                skip_peeling=False,
                qwen_model=str(payload.get("qwen_model") or "Qwen/Qwen-Image-Layered"),
                qwen_layers=str(payload.get("qwen_layers") or "3,4,6,8"),
                qwen_resolution=int(payload.get("qwen_resolution") or 640),
                qwen_steps=int(payload.get("qwen_steps") or 10),
                qwen_device=str(payload.get("qwen_device") or "cuda"),
                qwen_dtype=str(payload.get("qwen_dtype") or "bfloat16"),
                qwen_offload=str(payload.get("qwen_offload") or "sequential"),
                qwen_hybrid_modes=str(payload.get("qwen_hybrid_modes") or "preserve,reorder"),
                qwen_merge_external_layers=bool(payload.get("qwen_merge_external_layers", False)),
                limit=1,
                skip_existing=False,
                dry_run=False,
            )
            run_frontier_comparison(frontier_args)
            summary_path = frontier_root / "frontier_summary.json"
            summary = _read_json(summary_path)
            best_by_image = summary.get("best_by_image") or []
            if not best_by_image:
                raise RuntimeError("Frontier comparison produced no successful candidate selection")
            frontier_selection = dict(best_by_image[0])
            output_dir = _resolve_run_path(frontier_selection.get("run_dir"))
            if output_dir is None or not output_dir.exists():
                raise RuntimeError("Selected frontier run directory is missing")
            manifest_path = output_dir / "manifest.json"
            metrics_path = output_dir / "metrics.json"
            dalg_path = output_dir / "dalg_manifest.json"
            return {
                "status": "ok",
                "mode": mode,
                "input_filename": filename,
                "output_dir": _display_path(output_dir),
                "urls": {
                    "output_dir": _workspace_url(output_dir),
                    "summary": _workspace_url(summary_path),
                    "selected_best": _workspace_url(frontier_root / input_path.stem / "best_decomposition.json"),
                    "manifest": _workspace_url(manifest_path) if manifest_path.exists() else None,
                    "metrics": _workspace_url(metrics_path) if metrics_path.exists() else None,
                    "dalg": _workspace_url(dalg_path) if dalg_path.exists() else None,
                },
                "summary_metrics": _collect_summary_metrics(output_dir, mode, frontier_selection=frontier_selection),
                "previews": _collect_previews(output_dir, "run"),
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    output_dir = Path(out.output_dir)
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    dalg_path = output_dir / "dalg_manifest.json"
    return {
        "status": "ok",
        "mode": mode,
        "input_filename": filename,
        "output_dir": _display_path(output_dir),
        "urls": {
            "output_dir": _workspace_url(output_dir),
            "manifest": _workspace_url(manifest_path) if manifest_path.exists() else None,
            "metrics": _workspace_url(metrics_path) if metrics_path.exists() else None,
            "dalg": _workspace_url(dalg_path) if dalg_path.exists() else None,
        },
        "summary_metrics": _collect_summary_metrics(output_dir, mode, frontier_selection=frontier_selection),
        "previews": _collect_previews(output_dir, mode),
    }


class LayerForgeWebUIHandler(BaseHTTPRequestHandler):
    server_version = "LayerForgeWebUI/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path) -> None:
        content = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (mime or "application/octet-stream") + "; charset=utf-8" if (mime or "").startswith(("text/", "application/json", "application/javascript")) else (mime or "application/octet-stream"))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/runtime":
            self._send_json(
                {
                    "available": True,
                    "mode": "local",
                    "repo_root": str(REPO_ROOT),
                    "default_url": "/webui.html",
                }
            )
            return
        if parsed.path == "/api/project-summary":
            self._send_json(build_project_site_payload(REPO_ROOT))
            return
        if parsed.path.startswith("/workspace/"):
            try:
                file_path = _resolve_workspace_path(parsed.path)
            except PermissionError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
                return
            if not file_path.exists() or not file_path.is_file():
                self._send_json({"error": f"Missing file: {file_path}"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_file(file_path)
            return

        rel_path = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
        file_path = (DOCS_ROOT / rel_path).resolve()
        if DOCS_ROOT.resolve() not in file_path.parents and file_path != DOCS_ROOT.resolve():
            self._send_json({"error": "Requested path escapes docs root"}, status=HTTPStatus.FORBIDDEN)
            return
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.exists():
            self._send_json({"error": f"Missing file: {rel_path}"}, status=HTTPStatus.NOT_FOUND)
            return
        if file_path.name == "webui.html":
            html = _inject_runtime_marker(file_path.read_text(encoding="utf-8")).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        self._send_file(file_path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self._send_json({"error": "Unsupported endpoint"}, status=HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        try:
            result = run_webui_job(REPO_ROOT, payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(result, status=HTTPStatus.OK)


def serve_webui(host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = False) -> int:
    httpd = ThreadingHTTPServer((host, port), LayerForgeWebUIHandler)
    url = f"http://{host}:{port}/webui.html"
    print(f"LayerForge project site: http://{host}:{port}/")
    print(f"LayerForge local web UI: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
