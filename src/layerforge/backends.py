from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shlex
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Literal


BackendKind = Literal[
    "segmentation",
    "depth",
    "matting",
    "inpainting",
    "intrinsics",
    "amodal",
    "generative_layer",
]


@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    version: str | None
    device: str
    capabilities: tuple[str, ...]
    deterministic: bool
    input_schema: dict[str, str]
    output_schema: dict[str, str]
    confidence_outputs: tuple[str, ...] = ()
    available: bool = True
    fallback: str | None = None
    warning: str | None = None
    required_packages: tuple[str, ...] = ()
    kind: BackendKind = "segmentation"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SegmentationBackend(BackendSpec):
    kind: BackendKind = "segmentation"


@dataclass(frozen=True, slots=True)
class DepthBackend(BackendSpec):
    kind: BackendKind = "depth"


@dataclass(frozen=True, slots=True)
class MattingBackend(BackendSpec):
    kind: BackendKind = "matting"


@dataclass(frozen=True, slots=True)
class InpaintingBackend(BackendSpec):
    kind: BackendKind = "inpainting"


@dataclass(frozen=True, slots=True)
class IntrinsicsBackend(BackendSpec):
    kind: BackendKind = "intrinsics"


@dataclass(frozen=True, slots=True)
class AmodalBackend(BackendSpec):
    kind: BackendKind = "amodal"


@dataclass(frozen=True, slots=True)
class GenerativeLayerBackend(BackendSpec):
    kind: BackendKind = "generative_layer"


@dataclass(frozen=True, slots=True)
class BackendRegistry:
    segmentation: tuple[SegmentationBackend, ...]
    depth: tuple[DepthBackend, ...]
    matting: tuple[MattingBackend, ...]
    inpainting: tuple[InpaintingBackend, ...]
    intrinsics: tuple[IntrinsicsBackend, ...]
    amodal: tuple[AmodalBackend, ...]
    generative_layer: tuple[GenerativeLayerBackend, ...]

    def by_kind(self, kind: BackendKind) -> tuple[BackendSpec, ...]:
        return tuple(getattr(self, kind))

    def all(self) -> tuple[BackendSpec, ...]:
        return (
            *self.segmentation,
            *self.depth,
            *self.matting,
            *self.inpainting,
            *self.intrinsics,
            *self.amodal,
            *self.generative_layer,
        )

    def to_json(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "segmentation": [backend.to_json() for backend in self.segmentation],
            "depth": [backend.to_json() for backend in self.depth],
            "matting": [backend.to_json() for backend in self.matting],
            "inpainting": [backend.to_json() for backend in self.inpainting],
            "intrinsics": [backend.to_json() for backend in self.intrinsics],
            "amodal": [backend.to_json() for backend in self.amodal],
            "generative_layer": [backend.to_json() for backend in self.generative_layer],
        }


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _module_version(module: str, package: str | None = None) -> str | None:
    if package:
        version = package_version(package)
        if version is not None:
            return version
    try:
        imported = __import__(module)
    except Exception:
        return None
    return str(getattr(imported, "__version__", "")) or None


def _missing_warning(name: str, packages: tuple[str, ...], fallback: str | None) -> str:
    package_text = ", ".join(packages) if packages else "external backend"
    suffix = f"; falling back to {fallback}" if fallback else ""
    return f"{name} unavailable: missing {package_text}{suffix}"


def _not_configured_warning(name: str, fallback: str | None) -> str:
    suffix = f"; falling back to {fallback}" if fallback else ""
    return f"{name} unavailable: no external command configured{suffix}"


def _availability(
    name: str,
    modules: tuple[str, ...],
    packages: tuple[str, ...],
    fallback: str | None,
    *,
    external_command: str | None = None,
    env_key: str | None = None,
) -> tuple[bool, str | None]:
    missing = [module for module in modules if not module_available(module)]
    if external_command:
        try:
            parts = shlex.split(external_command)
        except ValueError:
            parts = []
        executable = parts[0] if parts else external_command
        if shutil.which(executable) is None:
            missing.append(executable)
    if env_key and not os.environ.get(env_key):
        missing.append(env_key)
    if missing:
        return False, _missing_warning(name, tuple(missing or packages), fallback)
    return True, None


def resolve_device(device: str = "auto") -> dict[str, Any]:
    requested = str(device or "auto")
    info: dict[str, Any] = {
        "requested": requested,
        "resolved": "cpu" if requested == "auto" else requested,
        "torch_available": module_available("torch"),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }
    if not info["torch_available"]:
        return info
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_available
        info["cuda_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
        if requested == "auto":
            info["resolved"] = "cuda" if cuda_available else "cpu"
        if cuda_available:
            info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info


def _hf_version() -> str | None:
    return package_version("transformers")


def build_backend_registry(config: dict[str, Any] | None = None, device: str = "auto") -> BackendRegistry:
    cfg = config or {}
    resolved_device = str(resolve_device(device).get("resolved", "cpu"))
    seg_model_cfg = cfg.get("segmentation", {}).get("model", {})
    depth_model_cfg = cfg.get("depth", {}).get("model", {})
    matting_cfg = cfg.get("matting", {})
    transparent_cfg = cfg.get("transparent", {})
    inpainting_cfg = cfg.get("inpainting", {})
    intrinsics_cfg = cfg.get("intrinsics", {})

    mask2former_ok, mask2former_warning = _availability("mask2former", ("transformers",), ("transformers",), "classical")
    grounded_ok, grounded_warning = _availability("grounded_sam2", ("torch", "transformers"), ("torch", "transformers"), "classical")
    gemini_ok, gemini_warning = _availability("gemini", (), (), "classical", env_key="GEMINI_API_KEY")

    segmentation = (
        SegmentationBackend(
            name="classical",
            version="grid-slic-heuristic",
            device="cpu",
            capabilities=("semantic_groups", "background_complement", "deterministic_grid"),
            deterministic=True,
            input_schema={"image": "RGB uint8 ndarray"},
            output_schema={"segments": "list[Segment] with boolean masks"},
            confidence_outputs=("score",),
            fallback=None,
        ),
        SegmentationBackend(
            name="mask2former",
            version=str(seg_model_cfg.get("mask2former", "facebook/mask2former-swin-large-coco-panoptic")),
            device=resolved_device,
            capabilities=("panoptic_masks", "semantic_labels", "stuff_object_regions"),
            deterministic=False,
            input_schema={"image": "PIL RGB image"},
            output_schema={"segments": "list[Segment] with model masks"},
            confidence_outputs=("score",),
            available=mask2former_ok,
            fallback="classical",
            warning=mask2former_warning,
            required_packages=("transformers",),
        ),
        SegmentationBackend(
            name="grounded_sam2",
            version=f"{seg_model_cfg.get('grounding_dino', 'IDEA-Research/grounding-dino-base')} + {seg_model_cfg.get('sam2', 'facebook/sam2.1-hiera-large')}",
            device=resolved_device,
            capabilities=("open_vocabulary_boxes", "sam2_masks", "promptable"),
            deterministic=False,
            input_schema={"image": "PIL RGB image", "prompts": "list[str]"},
            output_schema={"segments": "list[Segment] with refined masks"},
            confidence_outputs=("box_score", "sam_iou_score"),
            available=grounded_ok,
            fallback="classical",
            warning=grounded_warning,
            required_packages=("torch", "transformers"),
        ),
        SegmentationBackend(
            name="gemini",
            version=str(seg_model_cfg.get("gemini", "gemini-3-flash-preview")),
            device="remote",
            capabilities=("open_vocabulary_labels", "semantic_masks", "promptable"),
            deterministic=False,
            input_schema={"image": "PIL RGB image", "prompt": "Gemini JSON segmentation prompt"},
            output_schema={"items": "JSON labels, boxes, base64 masks"},
            confidence_outputs=(),
            available=gemini_ok,
            fallback="classical",
            warning=gemini_warning,
        ),
    )

    depth_anything_ok, depth_anything_warning = _availability("depth_anything_v2", ("transformers",), ("transformers",), "geometric_luminance")
    depth_pro_ok, depth_pro_warning = _availability("depth_pro", ("torch", "transformers"), ("torch", "transformers"), "geometric_luminance")
    marigold_ok, marigold_warning = _availability("marigold", ("torch", "diffusers"), ("torch", "diffusers"), "geometric_luminance")
    moge_ok, moge_warning = _availability("moge", ("moge",), ("moge",), "ensemble")
    depth = (
        DepthBackend(
            name="geometric_luminance",
            version="heuristic-v1",
            device="cpu",
            capabilities=("relative_depth", "edge_smoothing"),
            deterministic=True,
            input_schema={"image": "RGB uint8 ndarray"},
            output_schema={"depth": "float32 HxW normalized depth"},
            confidence_outputs=(),
        ),
        DepthBackend(
            name="depth_anything_v2",
            version=str(depth_model_cfg.get("depth_anything_v2", "depth-anything/Depth-Anything-V2-Large-hf")),
            device=resolved_device,
            capabilities=("relative_depth", "hf_pipeline"),
            deterministic=False,
            input_schema={"image": "PIL RGB image"},
            output_schema={"depth": "float32 HxW normalized depth", "raw_depth": "model depth"},
            confidence_outputs=(),
            available=depth_anything_ok,
            fallback="geometric_luminance",
            warning=depth_anything_warning,
            required_packages=("transformers",),
        ),
        DepthBackend(
            name="depth_pro",
            version=str(depth_model_cfg.get("depth_pro", "apple/DepthPro-hf")),
            device=resolved_device,
            capabilities=("metric_depth", "hf_depthpro"),
            deterministic=False,
            input_schema={"image": "PIL RGB image"},
            output_schema={"depth": "float32 HxW normalized depth", "raw_depth": "metric depth"},
            confidence_outputs=(),
            available=depth_pro_ok,
            fallback="geometric_luminance",
            warning=depth_pro_warning,
            required_packages=("torch", "transformers"),
        ),
        DepthBackend(
            name="marigold",
            version=str(depth_model_cfg.get("marigold", "prs-eth/marigold-depth-v1-1")),
            device=resolved_device,
            capabilities=("diffusion_depth", "relative_depth"),
            deterministic=False,
            input_schema={"image": "PIL RGB image"},
            output_schema={"depth": "float32 HxW normalized depth"},
            confidence_outputs=(),
            available=marigold_ok,
            fallback="geometric_luminance",
            warning=marigold_warning,
            required_packages=("torch", "diffusers"),
        ),
        DepthBackend(
            name="ensemble",
            version="median-rank-fusion",
            device=resolved_device,
            capabilities=("fallback_fusion", "edge_smoothing", "multi_backend"),
            deterministic=False,
            input_schema={"image": "PIL RGB image and RGB ndarray"},
            output_schema={"depth": "float32 HxW normalized fused depth", "errors": "per-backend warnings"},
            confidence_outputs=("backend_success_count",),
            fallback="geometric_luminance",
        ),
        DepthBackend(
            name="moge",
            version=str(depth_model_cfg.get("moge", "optional")),
            device=resolved_device,
            capabilities=("point_map", "geometry_depth", "optional_registry_backend"),
            deterministic=False,
            input_schema={"image": "RGB image"},
            output_schema={"depth": "float32 HxW", "point_map": "optional HxWx3 geometry"},
            confidence_outputs=("model_confidence",),
            available=moge_ok,
            fallback="ensemble",
            warning=moge_warning,
            required_packages=("moge",),
        ),
    )

    birefnet_cfg = {
        "model": matting_cfg.get("model") or transparent_cfg.get("model", "ZhengPeng7/BiRefNet-matting"),
    }
    birefnet_ok, birefnet_warning = _availability("birefnet", ("torch", "transformers"), ("torch", "transformers"), "heuristic")
    matting_anything_ok, matting_anything_warning = _availability("matting_anything", ("matting_anything",), ("matting_anything",), "heuristic")
    matting_external_command = str(matting_cfg.get("external_command", "")).strip()
    external_matting_ok = False
    external_matting_warning = _not_configured_warning("external_matting", "heuristic")
    if matting_external_command:
        external_matting_ok, external_matting_warning = _availability("external_matting", (), (), "heuristic", external_command=matting_external_command)
    matting = (
        MattingBackend(
            name="heuristic",
            version="distance-transform-alpha-v1",
            device="cpu",
            capabilities=("boundary_trimap", "depth_edge_preserving", "cpu_safe"),
            deterministic=True,
            input_schema={"image": "RGB uint8 ndarray", "mask": "boolean HxW support"},
            output_schema={"alpha": "float32 HxW in [0,1]"},
            confidence_outputs=("alpha_quality_score",),
        ),
        MattingBackend(
            name="birefnet",
            version=str(birefnet_cfg["model"]),
            device=resolved_device,
            capabilities=("model_alpha", "cropped_support_mask", "soft_foreground"),
            deterministic=False,
            input_schema={"image": "RGB uint8 ndarray", "support_mask": "boolean HxW"},
            output_schema={"alpha": "float32 HxW in [0,1]", "metadata": "crop/model/device"},
            confidence_outputs=("alpha_confidence",),
            available=birefnet_ok,
            fallback="heuristic",
            warning=birefnet_warning,
            required_packages=("torch", "transformers"),
        ),
        MattingBackend(
            name="matting_anything",
            version=str(matting_cfg.get("matting_anything_model", "optional")),
            device=resolved_device,
            capabilities=("promptable_matting", "trimap_or_mask_prompt"),
            deterministic=False,
            input_schema={"image": "RGB image", "trimap_or_prompt": "mask boundary prompt"},
            output_schema={"alpha": "float32 HxW in [0,1]"},
            confidence_outputs=("model_confidence",),
            available=matting_anything_ok,
            fallback="heuristic",
            warning=matting_anything_warning,
            required_packages=("matting_anything",),
        ),
        MattingBackend(
            name="external",
            version=matting_external_command or None,
            device="external",
            capabilities=("external_command",),
            deterministic=False,
            input_schema={"image": "path", "mask": "path"},
            output_schema={"alpha": "path or image"},
            confidence_outputs=(),
            available=external_matting_ok,
            fallback="heuristic",
            warning=external_matting_warning,
        ),
    )

    simple_lama_ok, simple_lama_warning = _availability("simple_lama", ("simple_lama_inpainting",), ("simple-lama-inpainting",), "telea")
    diffusion_ok, diffusion_warning = _availability("diffusion", ("torch", "diffusers"), ("torch", "diffusers"), "telea")
    inpaint_external_command = str(inpainting_cfg.get("external_command", "")).strip()
    external_inpaint_ok = False
    external_inpaint_warning = _not_configured_warning("external_inpainting", "telea")
    if inpaint_external_command:
        external_inpaint_ok, external_inpaint_warning = _availability("external_inpainting", (), (), "telea", external_command=inpaint_external_command)
    inpainting = (
        InpaintingBackend(
            name="telea",
            version=_module_version("cv2", "opencv-python"),
            device="cpu",
            capabilities=("background_completion", "cpu_safe", "deterministic"),
            deterministic=True,
            input_schema={"image": "RGB uint8 ndarray", "mask": "boolean HxW"},
            output_schema={"image": "RGB uint8 ndarray", "mask": "boolean HxW", "method": "str"},
            confidence_outputs=(),
        ),
        InpaintingBackend(
            name="simple_lama",
            version=package_version("simple-lama-inpainting"),
            device="cpu",
            capabilities=("lama_completion", "cpu_fallback"),
            deterministic=False,
            input_schema={"image": "PIL RGB image", "mask": "PIL L image"},
            output_schema={"image": "RGB uint8 ndarray"},
            confidence_outputs=(),
            available=simple_lama_ok,
            fallback="telea",
            warning=simple_lama_warning,
            required_packages=("simple-lama-inpainting",),
        ),
        InpaintingBackend(
            name="lama",
            version=str(inpainting_cfg.get("lama_model", "optional")),
            device=resolved_device,
            capabilities=("layer_aware_completion", "optional_checkpoint"),
            deterministic=False,
            input_schema={"image": "RGB image", "mask": "hidden/background mask"},
            output_schema={"image": "RGB image"},
            confidence_outputs=("completion_confidence",),
            available=simple_lama_ok,
            fallback="telea",
            warning=simple_lama_warning,
            required_packages=("simple-lama-inpainting",),
        ),
        InpaintingBackend(
            name="diffusion",
            version=str(inpainting_cfg.get("diffusion_model", "optional")),
            device=resolved_device,
            capabilities=("prompt_conditioned_completion", "content_preserving_prompt"),
            deterministic=False,
            input_schema={"image": "RGB image", "mask": "boolean HxW", "prompt": "optional str"},
            output_schema={"image": "RGB image"},
            confidence_outputs=("model_confidence",),
            available=diffusion_ok,
            fallback="telea",
            warning=diffusion_warning,
            required_packages=("torch", "diffusers"),
        ),
        InpaintingBackend(
            name="external",
            version=inpaint_external_command or None,
            device="external",
            capabilities=("external_command",),
            deterministic=False,
            input_schema={"image": "path", "mask": "path"},
            output_schema={"image": "path or image"},
            confidence_outputs=(),
            available=external_inpaint_ok,
            fallback="telea",
            warning=external_inpaint_warning,
        ),
    )

    iid_external_command = str(intrinsics_cfg.get("external_command", "")).strip()
    external_iid_ok = False
    external_iid_warning = _not_configured_warning("external_intrinsics", "retinex")
    if iid_external_command:
        external_iid_ok, external_iid_warning = _availability("external_intrinsics", (), (), "retinex", external_command=iid_external_command)
    intrinsic_model_ok, intrinsic_model_warning = _availability("intrinsic_model", ("torch",), ("torch",), "retinex")
    intrinsics = (
        IntrinsicsBackend(
            name="none",
            version="identity",
            device="cpu",
            capabilities=("disabled",),
            deterministic=True,
            input_schema={"image": "RGB uint8 ndarray"},
            output_schema={"albedo": "input RGB", "shading": "unit RGB"},
            confidence_outputs=("intrinsic_residual",),
        ),
        IntrinsicsBackend(
            name="retinex",
            version="single_scale_retinex",
            device="cpu",
            capabilities=("albedo_shading", "cpu_safe"),
            deterministic=True,
            input_schema={"image": "RGB uint8 ndarray"},
            output_schema={"albedo": "RGB uint8 ndarray", "shading": "RGB uint8 ndarray"},
            confidence_outputs=("intrinsic_residual",),
        ),
        IntrinsicsBackend(
            name="ordinal",
            version=str(intrinsics_cfg.get("ordinal_model", "optional")),
            device=resolved_device,
            capabilities=("ordinal_shading_constraints", "optional_checkpoint"),
            deterministic=False,
            input_schema={"image": "RGB image"},
            output_schema={"albedo": "RGB image", "shading": "RGB image"},
            confidence_outputs=("model_confidence", "intrinsic_residual"),
            available=intrinsic_model_ok,
            fallback="retinex",
            warning=intrinsic_model_warning,
            required_packages=("torch",),
        ),
        IntrinsicsBackend(
            name="intrinsic_model",
            version=str(intrinsics_cfg.get("intrinsic_model", "optional")),
            device=resolved_device,
            capabilities=("model_albedo_shading", "optional_checkpoint"),
            deterministic=False,
            input_schema={"image": "RGB image"},
            output_schema={"albedo": "RGB image", "shading": "RGB image"},
            confidence_outputs=("model_confidence", "intrinsic_residual"),
            available=intrinsic_model_ok,
            fallback="retinex",
            warning=intrinsic_model_warning,
            required_packages=("torch",),
        ),
        IntrinsicsBackend(
            name="external",
            version=iid_external_command or None,
            device="external",
            capabilities=("external_command",),
            deterministic=False,
            input_schema={"image": "path"},
            output_schema={"albedo": "path", "shading": "path"},
            confidence_outputs=("intrinsic_residual",),
            available=external_iid_ok,
            fallback="retinex",
            warning=external_iid_warning,
        ),
    )

    sameo_ok, sameo_warning = _availability("sameo", ("sameo",), ("sameo",), "heuristic")
    amodal_external_command = str(cfg.get("amodal", {}).get("external_command", "")).strip()
    external_amodal_ok = False
    external_amodal_warning = _not_configured_warning("external_amodal", "heuristic")
    if amodal_external_command:
        external_amodal_ok, external_amodal_warning = _availability("external_amodal", (), (), "heuristic", external_command=amodal_external_command)
    amodal = (
        AmodalBackend(
            name="none",
            version="disabled",
            device="cpu",
            capabilities=("visible_only",),
            deterministic=True,
            input_schema={"visible_mask": "boolean HxW"},
            output_schema={"amodal_mask": "None", "hidden_mask": "None"},
            confidence_outputs=(),
        ),
        AmodalBackend(
            name="heuristic",
            version="morphology-convex-hull-v1",
            device="cpu",
            capabilities=("mask_completion", "hidden_mask_estimate", "cpu_safe"),
            deterministic=True,
            input_schema={"visible_mask": "boolean HxW"},
            output_schema={"amodal_mask": "boolean HxW", "hidden_mask": "amodal - visible"},
            confidence_outputs=("hidden_area_ratio", "edge_continuity_score"),
        ),
        AmodalBackend(
            name="sameo",
            version=str(cfg.get("amodal", {}).get("sameo_model", "optional")),
            device=resolved_device,
            capabilities=("model_amodal_completion", "object_extent"),
            deterministic=False,
            input_schema={"image": "RGB image", "visible_mask": "boolean HxW"},
            output_schema={"amodal_mask": "boolean HxW", "confidence": "float"},
            confidence_outputs=("model_confidence", "hidden_area_ratio"),
            available=sameo_ok,
            fallback="heuristic",
            warning=sameo_warning,
            required_packages=("sameo",),
        ),
        AmodalBackend(
            name="external",
            version=amodal_external_command or None,
            device="external",
            capabilities=("external_command",),
            deterministic=False,
            input_schema={"image": "path", "visible_mask": "path"},
            output_schema={"amodal_mask": "path or mask"},
            confidence_outputs=("model_confidence",),
            available=external_amodal_ok,
            fallback="heuristic",
            warning=external_amodal_warning,
        ),
    )

    qwen_ok, qwen_warning = _availability("qwen_image_layered", ("torch", "diffusers", "transformers"), ("torch", "diffusers", "transformers"), None)
    layerdecomp_ok, layerdecomp_warning = _availability("layerdecomp", ("layerdecomp",), ("layerdecomp",), None)
    diffdecompose_ok, diffdecompose_warning = _availability("diffdecompose", ("diffdecompose",), ("diffdecompose",), None)
    rld_ok, rld_warning = _availability("rld", ("rld",), ("rld",), None)
    generative_layer = (
        GenerativeLayerBackend(
            name="qwen_image_layered",
            version=str(cfg.get("qwen", {}).get("model", "Qwen/Qwen-Image-Layered")),
            device=resolved_device,
            capabilities=("generative_rgba_layers", "external_layer_import", "qwen_enrichment"),
            deterministic=False,
            input_schema={"image": "RGB image", "layer_count": "int or list[int]"},
            output_schema={"layers": "RGBA images", "manifest": "optional external metadata"},
            confidence_outputs=("raw_layer_score",),
            available=qwen_ok,
            warning=qwen_warning,
            required_packages=("torch", "diffusers", "transformers"),
        ),
        GenerativeLayerBackend(
            name="layerdecomp",
            version="optional",
            device=resolved_device,
            capabilities=("adapter_candidate", "benchmark_comparison"),
            deterministic=False,
            input_schema={"image": "RGB image"},
            output_schema={"layers": "RGBA images"},
            confidence_outputs=("adapter_confidence",),
            available=layerdecomp_ok,
            warning=layerdecomp_warning,
            required_packages=("layerdecomp",),
        ),
        GenerativeLayerBackend(
            name="diffdecompose",
            version="optional",
            device=resolved_device,
            capabilities=("adapter_candidate", "benchmark_comparison"),
            deterministic=False,
            input_schema={"image": "RGB image"},
            output_schema={"layers": "RGBA images"},
            confidence_outputs=("adapter_confidence",),
            available=diffdecompose_ok,
            warning=diffdecompose_warning,
            required_packages=("diffdecompose",),
        ),
        GenerativeLayerBackend(
            name="rld",
            version="optional",
            device=resolved_device,
            capabilities=("adapter_candidate", "benchmark_comparison"),
            deterministic=False,
            input_schema={"image": "RGB image"},
            output_schema={"layers": "RGBA images"},
            confidence_outputs=("adapter_confidence",),
            available=rld_ok,
            warning=rld_warning,
            required_packages=("rld",),
        ),
    )

    return BackendRegistry(
        segmentation=segmentation,
        depth=depth,
        matting=matting,
        inpainting=inpainting,
        intrinsics=intrinsics,
        amodal=amodal,
        generative_layer=generative_layer,
    )
