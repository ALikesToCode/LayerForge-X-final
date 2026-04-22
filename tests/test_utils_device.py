from __future__ import annotations

from layerforge.utils import transformers_pipeline_device_index


def test_transformers_pipeline_device_index_handles_cuda_variants() -> None:
    assert transformers_pipeline_device_index("cpu") == -1
    assert transformers_pipeline_device_index("cuda") == 0
    assert transformers_pipeline_device_index("cuda:0") == 0
    assert transformers_pipeline_device_index("cuda:1") == 1
