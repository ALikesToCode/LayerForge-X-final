from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backends import BackendRegistry, build_backend_registry, module_available, package_version, resolve_device
from .config import load_config


REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("scipy", "scipy"),
    ("skimage", "scikit-image"),
    ("yaml", "PyYAML"),
    ("networkx", "networkx"),
    ("tqdm", "tqdm"),
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    hard_required: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoctorReport:
    status: str
    python: dict[str, Any]
    packages: dict[str, dict[str, Any]]
    device: dict[str, Any]
    paths: dict[str, CheckResult]
    backend_registry: BackendRegistry
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "python": self.python,
            "packages": self.packages,
            "device": self.device,
            "paths": {key: value.to_json() for key, value in self.paths.items()},
            "backend_registry": self.backend_registry.to_json(),
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
        }


def _check_writable_dir(path: str | Path, name: str) -> CheckResult:
    target = Path(path).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".layerforge_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return CheckResult(name=name, ok=True, detail=str(target), hard_required=True)
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=f"{target}: {type(exc).__name__}: {exc}", hard_required=True)


def collect_package_versions() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {
        "layerforge-x": {
            "module": "layerforge",
            "version": package_version("layerforge-x"),
            "available": module_available("layerforge"),
            "required": True,
        }
    }
    for module_name, package_name in REQUIRED_PACKAGES:
        rows[package_name] = {
            "module": module_name,
            "version": package_version(package_name),
            "available": module_available(module_name),
            "required": True,
        }
    for module_name, package_name in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("simple_lama_inpainting", "simple-lama-inpainting"),
        ("kornia", "kornia"),
        ("timm", "timm"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
    ):
        rows[package_name] = {
            "module": module_name,
            "version": package_version(package_name),
            "available": module_available(module_name),
            "required": False,
        }
    return rows


def build_doctor_report(
    *,
    config_path: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> DoctorReport:
    cfg = load_config(config_path, config_overrides)
    device_info = resolve_device(device)
    registry = build_backend_registry(cfg, device=device)
    packages = collect_package_versions()
    paths = {
        "cache_dir": _check_writable_dir(cache_dir or Path.home() / ".cache" / "layerforge", "cache_dir"),
        "output_dir": _check_writable_dir(output_dir or Path.cwd() / "runs", "output_dir"),
    }
    python_info = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "required": ">=3.10",
        "ok": sys.version_info >= (3, 10),
    }

    hard_failures: list[str] = []
    if not python_info["ok"]:
        hard_failures.append(f"python {python_info['version']} is below required >=3.10")
    for package_name, row in packages.items():
        if row["required"] and not row["available"]:
            hard_failures.append(f"required package missing: {package_name}")
    for check in paths.values():
        if check.hard_required and not check.ok:
            hard_failures.append(check.detail)

    seen_warnings: set[str] = set()
    warnings: list[str] = []
    for backend in registry.all():
        if backend.warning and backend.warning not in seen_warnings:
            seen_warnings.add(backend.warning)
            warnings.append(backend.warning)
    return DoctorReport(
        status="ok" if not hard_failures else "failed",
        python=python_info,
        packages=packages,
        device=device_info,
        paths=paths,
        backend_registry=registry,
        hard_failures=tuple(hard_failures),
        warnings=tuple(warnings),
    )


def render_doctor_text(report: DoctorReport) -> str:
    lines: list[str] = []
    lines.append("LayerForge doctor")
    lines.append(f"status: {report.status}")
    lines.append(f"python: {report.python['version']} ({report.python['executable']})")
    lines.append("")
    lines.append("required packages:")
    for package_name, row in sorted(report.packages.items()):
        if not row["required"]:
            continue
        version = row["version"] or "not installed"
        marker = "ok" if row["available"] else "missing"
        lines.append(f"  {marker:7} {package_name}: {version}")
    lines.append("")
    lines.append("optional packages:")
    for package_name, row in sorted(report.packages.items()):
        if row["required"]:
            continue
        version = row["version"] or "not installed"
        marker = "ok" if row["available"] else "missing"
        lines.append(f"  {marker:7} {package_name}: {version}")
    lines.append("")
    lines.append("device:")
    lines.append(f"  requested: {report.device.get('requested')}")
    lines.append(f"  resolved: {report.device.get('resolved')}")
    lines.append(f"  torch_available: {report.device.get('torch_available')}")
    lines.append(f"  cuda_available: {report.device.get('cuda_available')}")
    if report.device.get("cuda_device_name"):
        lines.append(f"  cuda_device_name: {report.device.get('cuda_device_name')}")
    lines.append("")
    lines.append("paths:")
    for key, check in report.paths.items():
        marker = "ok" if check.ok else "failed"
        lines.append(f"  {marker:7} {key}: {check.detail}")
    lines.append("")
    lines.append("backends:")
    for kind in ("segmentation", "depth", "matting", "inpainting", "intrinsics", "amodal", "generative_layer"):
        lines.append(f"  {kind}:")
        for backend in report.backend_registry.by_kind(kind):  # type: ignore[arg-type]
            marker = "ok" if backend.available else "fallback"
            version = backend.version or "unspecified"
            fallback = f", fallback={backend.fallback}" if backend.fallback else ""
            lines.append(f"    {marker:8} {backend.name} ({version}, device={backend.device}{fallback})")
            if backend.warning:
                lines.append(f"      warning: {backend.warning}")
    if report.hard_failures:
        lines.append("")
        lines.append("hard failures:")
        for failure in report.hard_failures:
            lines.append(f"  - {failure}")
    return "\n".join(lines)


def doctor_exit_code(report: DoctorReport) -> int:
    return 1 if report.hard_failures else 0


def doctor_json(report: DoctorReport) -> str:
    return json.dumps(report.to_json(), indent=2, sort_keys=True)
