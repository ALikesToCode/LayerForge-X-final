from __future__ import annotations

from layerforge.segment import _post_process_grounding_dino


class NewProcessor:
    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids=None,
        threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes=None,
        text_labels=None,
    ):
        self.last_call = {
            "outputs": outputs,
            "input_ids": input_ids,
            "threshold": threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
            "text_labels": text_labels,
        }
        return [{"boxes": [], "labels": [], "scores": []}]


class OldProcessor:
    def __init__(self) -> None:
        self.last_call: dict[str, object] | None = None

    def post_process_grounded_object_detection(
        self,
        outputs,
        input_ids=None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes=None,
    ):
        self.last_call = {
            "outputs": outputs,
            "input_ids": input_ids,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "target_sizes": target_sizes,
        }
        return [{"boxes": [], "labels": [], "scores": []}]


def test_post_process_grounding_dino_uses_new_threshold_signature() -> None:
    proc = NewProcessor()

    det = _post_process_grounding_dino(
        proc,
        outputs="outputs",
        input_ids="ids",
        box_threshold=0.42,
        text_threshold=0.11,
        target_sizes=[(10, 20)],
    )

    assert det == {"boxes": [], "labels": [], "scores": []}
    assert proc.last_call is not None
    assert proc.last_call["threshold"] == 0.42
    assert "box_threshold" not in proc.last_call


def test_post_process_grounding_dino_falls_back_to_old_box_threshold_signature() -> None:
    proc = OldProcessor()

    det = _post_process_grounding_dino(
        proc,
        outputs="outputs",
        input_ids="ids",
        box_threshold=0.37,
        text_threshold=0.09,
        target_sizes=[(10, 20)],
    )

    assert det == {"boxes": [], "labels": [], "scores": []}
    assert proc.last_call is not None
    assert proc.last_call["box_threshold"] == 0.37
