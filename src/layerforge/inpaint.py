from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .utils import optional_import


def inpaint_background(rgb: np.ndarray, mask: np.ndarray, cfg: dict[str, Any], device: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    method = str(cfg.get("method", "opencv_telea")).lower()
    m = mask.astype(bool)
    if not m.any():
        return rgb.copy(), m, "none"
    if method in {"opencv", "opencv_telea", "telea"}:
        cv2 = optional_import("cv2")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, (m.astype(np.uint8) * 255), float(cfg.get("radius", 5)), cv2.INPAINT_TELEA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), m, "opencv_telea"
    if method in {"lama", "simple_lama"}:
        try:
            from simple_lama_inpainting import SimpleLama
        except Exception as exc:
            raise RuntimeError("LaMa inpainting requires simple-lama-inpainting") from exc
        lama = SimpleLama()
        out = lama(Image.fromarray(rgb), Image.fromarray((m.astype(np.uint8) * 255), mode="L"))
        return np.asarray(out.convert("RGB"), dtype=np.uint8), m, "lama"
    raise ValueError(f"Unknown inpainting method: {method}")
