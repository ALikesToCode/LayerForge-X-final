from __future__ import annotations

GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "person": ("person", "man", "woman", "child", "boy", "girl", "face", "hair", "human"),
    "animal": ("animal", "dog", "cat", "bird", "horse", "cow", "sheep", "elephant", "bear", "zebra", "giraffe"),
    "vehicle": ("car", "truck", "bus", "train", "motorcycle", "bicycle", "boat", "airplane", "vehicle"),
    "furniture": ("chair", "table", "sofa", "couch", "bed", "cabinet", "desk", "bench", "stool", "shelf"),
    "plant": ("plant", "tree", "flower", "grass", "potted", "bush"),
    "sky": ("sky", "cloud"),
    "road": ("road", "street", "sidewalk", "path", "railroad", "runway"),
    "ground": ("ground", "sand", "snow", "field", "floor", "earth", "terrain"),
    "building": ("building", "house", "skyscraper", "wall", "window", "door", "roof", "bridge"),
    "water": ("water", "sea", "river", "lake", "ocean"),
    "stuff": ("curtain", "rug", "blanket", "cloth", "mountain", "hill", "background"),
}

BACKGROUND_GROUPS = {"sky", "road", "ground", "building", "water", "stuff", "background"}
THING_GROUPS = {"person", "animal", "vehicle", "furniture", "plant", "object"}


def label_to_group(label: str) -> str:
    low = str(label).lower()
    for group, keys in GROUP_KEYWORDS.items():
        if any(k in low for k in keys):
            return group
    return "object"
